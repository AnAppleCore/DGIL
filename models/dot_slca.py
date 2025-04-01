import logging
import os
import time

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.augmentation.auto import RandAugment, TrivialAugment
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.distributions import *
from utils.inc_net import SLCANet
from utils.losses import sup_con

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args:dict):
        super().__init__(args)
        self._network = SLCANet(args, pretrained=True)
        self.batch_size = args['batch_size']
        self.epochs = args['epochs']
        self.lrate = args['lrate']
        self.lrate_decay = args['lrate_decay']
        self.weight_decay = args['weight_decay']
        self.milestones = args['milestones']
        self.bcb_lrscale = args.get('bcb_lrscale', 1.0/100)
        self.fix_bcb = self.bcb_lrscale == 0
        
        self.ca_epochs = args['ca_epochs']
        self.logit_norm = args.get('ca_with_logit_norm', None)
        self.save_before_ca = args.get('save_before_ca', False)

        self.dot_epochs = args.get('dot_epochs', 0)
        self.domain_centroids = args.get('domain_centroids', 32)
        self.dom_loss_weight = args.get('dom_loss_weight', 1.0)

        self.orth_loss_weight = args.get('orth_loss_weight', 0.0)
        
        # augmentation used for single-source DG
        self.use_rand_aug = args.get('use_rand_aug', False)
        self.rand_aug = K.AugmentationSequential(RandAugment(n=2, m=10))
        self.trivial_aug = K.AugmentationSequential(TrivialAugment())

        self.args = args
        self.seed = args['seed']
        self.task_sizes = []

        self._cur_domain = 0
        self.cls_to_task_id = {}
        self.cls_to_domain_id = {}
        self.domain_id_to_cls = {}
        self.cls_dists = {}
        self.prototypes = None
        self.prototypes_domain_id = None
        # dist type: mean_cov, mean_var, multi_cen
        self.dist_type = args.get('dist_type', 'mean_cov')

        self.tsne_visualize = args.get('tsne_visualize', False)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        # self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}'.format(self.seed), head_only=self.fix_bcb)
        self._network.fc.recall()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.topk = min(self.topk, self._total_classes)
        self._network.update_fc(data_manager.get_task_size(self._cur_task))

        try:
            self._cur_domain = data_manager.get_cur_domain(self._cur_task)
        except:
            self._cur_domain = 0
        for c_id in range(self._known_classes, self._total_classes):
            self.cls_to_task_id[c_id] = self._cur_task
            self.cls_to_domain_id[c_id] = self._cur_domain
            if self._cur_domain not in self.domain_id_to_cls:
                self.domain_id_to_cls[self._cur_domain] = []
            self.domain_id_to_cls[self._cur_domain].append(c_id)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        self._network.to(self._device)

        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train',
                                                  appendent=[])
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        dset_name = data_manager.dataset_name.lower()

        self.train_loader = DataLoader(train_dset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self._stage1_training(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # CA
        self._network.fc.backup()
        # if self.save_before_ca:
            # self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}_before_ca'.format(self.seed), head_only=self.fix_bcb)
        
        self._compute_distributions(data_manager)
        if len(self.domain_id_to_cls.keys()) > 1 and self.dot_epochs > 0:
            self._stage2_domain_transformation()
        if self._cur_task > 0 and self.ca_epochs > 0:
            self._stage3_compact_classifier()
        

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            losses_orth = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                if self.use_rand_aug:
                    inputs = self.rand_aug(inputs)

                output = self._network(inputs, bcb_no_grad=self.fix_bcb)
                logits = output['logits']
                features = output['pre_logits']
                cur_targets = torch.where(targets-self._known_classes>=0, targets-self._known_classes, -100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                if self.orth_loss_weight > 0:
                    # loss_orth = self._compute_orth_loss(features)
                    loss_orth = self._compute_orth_loss(features, cur_targets)
                    loss += self.orth_loss_weight * loss_orth
                    losses_orth += loss_orth.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()

            train_acc = self._compute_accuracy(self._network, train_loader)
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_orth {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    losses_orth / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_orth {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    losses_orth / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _stage1_training(self, train_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''
        base_params = self._network.backbone.parameters()
        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        head_scale = 1.
        if not self.fix_bcb:
            base_params = {'params': base_params, 'lr': self.lrate*self.bcb_lrscale, 'weight_decay': self.weight_decay}
            base_fc_params = {'params': base_fc_params, 'lr': self.lrate*head_scale, 'weight_decay': self.weight_decay}
            network_params = [base_params, base_fc_params]
        else:
            for p in base_params:
                p.requires_grad = False
            network_params = [{'params': base_fc_params, 'lr': self.lrate*head_scale, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._run(train_loader, test_loader, optimizer, scheduler)


    def _stage2_domain_transformation(self):
        run_epochs = self.dot_epochs
        crct_num = self._total_classes
        self._network.reset_domain_tsf_clf(
            # nb_classes=self._total_classes, num_domains=self.data_manager.num_domains
            nb_classes=512, num_domains=512
        )
        param_list = {}
        for n, p in self._network.named_parameters():
            if "domain" in n or "class" in n:
                p.requires_grad = True
                param_list[n] = p
        # logging.info(f"DoT trainnable params: {param_list.keys()}")
        network_params = [{'params': param_list.values(), 'lr': self.lrate, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.eval()
        for epoch in range(run_epochs):
            losses = 0.
            losses_cls, losses_dom = 0., 0.

            sampled_data = []
            sampled_label = []
            sampled_domain_id = []
            num_sampled_pcls = 256

            for c_id in range(crct_num):
                t_id = self.cls_to_task_id[c_id]
                d_id = self.cls_to_domain_id[c_id]

                # cls_mean = torch.tensor(self._class_means_slca[c_id], dtype=torch.float64).to(self._device)
                # cls_cov = self._class_covs_slca[c_id].to(self._device)
                
                # m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                # sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))

                cls_dist: BaseDist = self.cls_dists[c_id]
                sampled_data_single = cls_dist.generate(num_sampled_pcls)

                sampled_data.append(sampled_data_single)                
                sampled_label.extend([c_id]*num_sampled_pcls)
                sampled_domain_id.extend([d_id]*num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            sampled_domain_id = torch.tensor(sampled_domain_id).long().to(self._device)

            sf_indexes = torch.randperm(sampled_data.size(0))
            inputs = sampled_data[sf_indexes]
            targets = sampled_label[sf_indexes]
            domain_id = sampled_domain_id[sf_indexes]

            for _iter in range(crct_num):
                inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                tgt = targets[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                did = domain_id[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]

                ptp_id = np.random.choice(
                    len(self.prototypes), size=num_sampled_pcls, replace=True
                )
                ptp_inps = self.prototypes[ptp_id]
                ptp_dids = self.prototypes_domain_id[ptp_id]
                ptp_inps = torch.tensor(ptp_inps).float().to(self._device)
                ptp_dids = torch.tensor(ptp_dids).long().to(self._device)

                fake_inps = self._network.domain_tsf(inp, ptp_inps)

                all_inps = torch.cat([inp, fake_inps], dim=0)
                all_tgts = torch.cat([tgt, tgt], dim=0)
                all_dids = torch.cat([did, ptp_dids], dim=0)

                all_class_outputs = self._network.class_clf(all_inps)
                cls_loss = sup_con(features=all_class_outputs, labels=all_tgts)

                all_domain_outputs = self._network.domain_clf(all_inps)
                dom_loss = sup_con(features=all_domain_outputs, labels=all_dids)

                loss = cls_loss + dom_loss * self.dom_loss_weight

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_cls += cls_loss.item()
                losses_dom += dom_loss.item()

            scheduler.step()

            info = 'DOT Task {} => Loss {:.3f}, Cls_loss {:.3f}, Dom_loss {:.3f}'.format(
                self._cur_task, losses/self._total_classes, losses_cls/self._total_classes, losses_dom/self._total_classes)
            logging.info(info)


    def _stage3_compact_classifier(self):
        for p in self._network.fc.parameters():
            p.requires_grad=True
            
        run_epochs = self.ca_epochs
        crct_num = self._total_classes    
        param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': self.lrate,
                           'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4], gamma=lrate_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.eval()
        for epoch in range(run_epochs):
            losses = 0.

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256
        
            for c_id in range(crct_num):
                t_id = self.cls_to_task_id[c_id]
                decay = (t_id+1)/(self._cur_task+1)*0.1
                # cls_mean = torch.tensor(self._class_means_slca[c_id], dtype=torch.float64).to(self._device)*(0.9+decay)
                # cls_cov = self._class_covs_slca[c_id].to(self._device)
                
                # m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                # sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))

                cls_dist: BaseDist = self.cls_dists[c_id]
                sampled_data_single = cls_dist.generate(num_sampled_pcls, decay=decay)

                sampled_data.append(sampled_data_single)                
                sampled_label.extend([c_id]*num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)

            inputs = sampled_data
            targets= sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(crct_num):
                inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                tgt = targets[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]

                if len(self.domain_id_to_cls.keys()) > 1 and self.dot_epochs > 0:
                    ptp_id = np.random.choice(
                        len(self.prototypes), size=num_sampled_pcls, replace=True
                    )
                    ptp_inps = self.prototypes[ptp_id]
                    ptp_dids = self.prototypes_domain_id[ptp_id]
                    ptp_inps = torch.tensor(ptp_inps).float().to(self._device)
                    ptp_dids = torch.tensor(ptp_dids).long().to(self._device)

                    fake_inps = self._network.domain_tsf(inp, ptp_inps)

                    inp = torch.cat([inp, fake_inps], dim=0)
                    tgt = torch.cat([tgt, tgt], dim=0)

                outputs = self._network(inp, bcb_no_grad=True, fc_only=True)
                logits = outputs['logits']

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task+1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                        
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)

                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, losses/self._total_classes, test_acc)
            logging.info(info)


    def _compute_distributions(self, data_manager):
        # if hasattr(self, '_class_means_slca') and self._class_means_slca is not None:
        #     ori_classes = self._class_means_slca.shape[0]
        #     assert ori_classes==self._known_classes
        #     new_class_means_slca = np.zeros((self._total_classes, self.feature_dim))
        #     new_class_means_slca[:self._known_classes] = self._class_means_slca
        #     self._class_means_slca = new_class_means_slca
        #     new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
        #     new_class_cov[:self._known_classes] = self._class_covs_slca
        #     self._class_covs_slca = new_class_cov
        # else:
        #     self._class_means_slca = np.zeros((self._total_classes, self.feature_dim))
        #     self._class_covs_slca = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
        
        all_features = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            if self.dot_epochs > 0:
                vectors, features, _ = self._extract_layerwise_vectors(idx_loader)
                all_features.append(features)
            else:
                vectors, _ = self._extract_vectors(idx_loader)

            # class_mean = np.mean(vectors, axis=0)
            # class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(class_mean.shape[-1])*1e-4
            # self._class_means_slca[class_idx, :] = class_mean
            # self._class_covs_slca[class_idx, ...] = class_cov

            vectors = torch.tensor(vectors, dtype=torch.float64).to(self._device)
            if self.dist_type == 'mean_cov':
                cls_dist = CovarianceDist(feature_dim=vectors.shape[-1], device=self._device)
                cls_mean = torch.mean(vectors, dim=0)
                cls_cov = torch.cov(vectors.T)
                cls_dist.init_from(mean=cls_mean, cov=cls_cov, num_samples=vectors.shape[0])
            elif self.dist_type == 'mean_var':
                cls_dist = VarianceDist(feature_dim=vectors.shape[-1], device=self._device)
                cls_mean = torch.mean(vectors, dim=0)
                cls_var = torch.var(vectors, dim=0)
                cls_dist.init_from(mean=cls_mean, var=cls_var, num_samples=vectors.shape[0])
            elif self.dist_type == 'multi_cen':
                n_centroids = min(10, vectors.shape[0])
                cls_dist = MultiCentroidDist(n_centroids=n_centroids, feature_dim=vectors.shape[-1], device=self._device)
                cls_dist.compute_centroids(vectors)
            else:
                raise NotImplementedError(f"Unsupported distribution type: {self.dist_type}")
            
            self.cls_dists[class_idx] = cls_dist

        logging.info('Compute distributions for classes {}-{}'.format(self._known_classes, self._total_classes))

        if self.dot_epochs > 0:
            all_features = np.concatenate(all_features, axis=0) # [num_samples, num_layers, feature_dim]
            all_features_mean = np.mean(all_features, axis=1) # [num_samples, feature_dim]
            kmeans = KMeans(n_clusters=self.domain_centroids).fit(all_features_mean)
            feature_centers = kmeans.cluster_centers_
            # find closest prototype for each center
            prototype_idx = []
            all_idx = np.arange(all_features_mean.shape[0])
            for i in range(self.domain_centroids):
                i_mask = (kmeans.labels_ == i)
                i_idx = all_idx[i_mask]
                dist = np.linalg.norm(all_features_mean[i_mask] - feature_centers[i], axis=1)
                prototype_idx.append(i_idx[np.argmin(dist)])

            if self.prototypes is None:
                self.prototypes = all_features[prototype_idx]
                self.prototypes_domain_id = np.zeros(self.domain_centroids, dtype=np.int32) + self._cur_domain
            else:
                self.prototypes = np.concatenate([self.prototypes, all_features[prototype_idx]], axis=0)
                self.prototypes_domain_id = np.concatenate([
                    self.prototypes_domain_id, np.zeros(self.domain_centroids, dtype=np.int32) + self._cur_domain
                ], axis=0)

        logging.info('Compute domain prototypes for domain {}'.format(self._cur_domain))

    
    def _extract_layerwise_vectors(self, loader):
        self._network.eval()
        vectors, features, targets = [], [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors, _features = self._network.module.extract_layerwise_vector(_inputs.to(self._device))
                else:
                    _vectors, _features = self._network.extract_layerwise_vector(_inputs.to(self._device))
                
                vectors.append(_vectors)
                features.append(_features)
                targets.append(_targets)

        vectors = np.concatenate(vectors)
        features = np.concatenate(features)
        targets = np.concatenate(targets)

        return vectors, features, targets


    # def _compute_orth_loss(self, features):
    #     if hasattr(self, '_class_means_slca') and self._class_means_slca is not None:
    #         sample_mean = []
    #         for c_id in range(self._known_classes):
    #             cls_mean = torch.tensor(self._class_means_slca[c_id], dtype=torch.float64)
    #             sample_mean.append(cls_mean)
    #         sample_mean = torch.stack(sample_mean, dim=0).to(self._device, non_blocking=True)
    #         M = torch.cat([sample_mean, features], dim=0).to(self._device, non_blocking=True)
    #         M = F.normalize(M, dim=1)
    #         sim = torch.matmul(M, M.t()) / 0.8
    #     else:
    #         features = F.normalize(features, dim=1)
    #         sim = torch.matmul(features, features.t()) / 0.8
    #     loss = F.cross_entropy(sim, torch.arange(sim.shape[0]).long().to(self._device))
    #     return loss


    def _compute_orth_loss(self, features, targets):
        # if hasattr(self, '_class_means_slca') and self._class_means_slca is not None:
        #     sample_mean = []
        #     sampled_label = []
        #     for c_id in range(self._known_classes):
        #         cls_mean = torch.tensor(self._class_means_slca[c_id], dtype=torch.float64)
        #         sample_mean.append(cls_mean)
        #         sampled_label.append(c_id)
        #     sample_mean = torch.stack(sample_mean, dim=0).to(self._device, non_blocking=True)
        #     sampled_label = torch.tensor(sampled_label).long().to(self._device, non_blocking=True)
        #     features = torch.cat([sample_mean, features], dim=0).to(self._device, non_blocking=True)
        #     targets = torch.cat([sampled_label, targets], dim=0).to(self._device, non_blocking=True)
        return sup_con(features=features, labels=targets)