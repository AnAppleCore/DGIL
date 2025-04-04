import logging

import kornia.augmentation as K
import numpy as np
import torch
from kornia.augmentation.auto import RandAugment, TrivialAugment
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import SLCANet

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
        
        self.args = args
        self.seed = args['seed']
        self.task_sizes = []
        self._cur_domain = 0
        self.cls_to_task_id = {}
        self.cls_to_domain_id = {}

        self.orth_loss_weight = args.get('orth_loss_weight', 0.0)
        self.mmda_loss_weight = args.get('mmda_loss_weight', 0.0)

        self.rand_aug = K.AugmentationSequential(RandAugment(n=2, m=10))
        self.trivial_aug = K.AugmentationSequential(TrivialAugment())

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
        if self._cur_task > 0 and self.ca_epochs > 0:
            self._stage2_compact_classifier(task_size)
        

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                outputs = self._network(inputs, bcb_no_grad=self.fix_bcb)
                logits, features = outputs['logits'], outputs['features']
                cur_targets = torch.where(targets-self._known_classes>=0, targets-self._known_classes, -100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                inputs_aug = self.trivial_aug(inputs)
                outptus_aug = self._network(inputs_aug, bcb_no_grad=self.fix_bcb)
                logits_aug, features_aug = outptus_aug['logits'], outptus_aug['features']
                loss += F.cross_entropy(logits_aug[:, self._known_classes:], cur_targets)

                if self.orth_loss_weight > 0:
                    loss += self._orth_loss(features, features_aug) * self.orth_loss_weight

                if self.mmda_loss_weight > 0:
                    loss += self._mmda_loss(torch.cat([features, features_aug], dim=0)) * self.mmda_loss_weight

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()

            train_acc = self._compute_accuracy(self._network, train_loader)
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
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


    def _stage2_compact_classifier(self, task_size):
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
                cls_mean = torch.tensor(self._class_means_slca[c_id], dtype=torch.float64).to(self._device)*(0.9+decay) # torch.from_numpy(self._class_means_slca[c_id]).to(self._device)
                cls_cov = self._class_covs_slca[c_id].to(self._device)
                
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
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
        if hasattr(self, '_class_means_slca') and self._class_means_slca is not None:
            ori_classes = self._class_means_slca.shape[0]
            assert ori_classes==self._known_classes
            new_class_means_slca = np.zeros((self._total_classes, self.feature_dim))
            new_class_means_slca[:self._known_classes] = self._class_means_slca
            self._class_means_slca = new_class_means_slca
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs_slca
            self._class_covs_slca = new_class_cov
        else:
            self._class_means_slca = np.zeros((self._total_classes, self.feature_dim))
            self._class_covs_slca = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
        
        all_vectors = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            all_vectors.append(vectors)

            # vectors = np.concatenate([vectors_aug, vectors])

            class_mean = np.mean(vectors, axis=0)
            # class_cov = np.cov(vectors.T)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(class_mean.shape[-1])*1e-4
            self._class_means_slca[class_idx, :] = class_mean
            self._class_covs_slca[class_idx, ...] = class_cov

        all_vectors = np.concatenate(all_vectors, axis=0)
        domain_mean = np.mean(all_vectors, axis=0)
        domain_cov = torch.cov(torch.tensor(all_vectors, dtype=torch.float64).T) + torch.eye(self.feature_dim) * 1e-4 

        if not hasattr(self, '_domain_means'):
            self._domain_means = []
            self._domain_covs = []
        self._domain_means.append(domain_mean)
        self._domain_covs.append(domain_cov)


    def _orth_loss(self, features:torch.Tensor, features_aug:torch.Tensor):
        if self._cur_task > 0:
            sample_data = []
            num_sampled_pcls = (self.batch_size // self._known_classes) + 1
            for c_id in range(self._known_classes):
                t_id = self.cls_to_task_id[c_id]
                decay = (t_id+1)/(self._cur_task+1)*0.1
                cls_mean = torch.tensor(self._class_means_slca[c_id], dtype=torch.float64).to(self._device)*(0.9+decay)
                cls_cov = self._class_covs_slca[c_id].to(self._device)
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sample_data.append(sampled_data_single)
            sample_data = torch.cat(sample_data, dim=0).float().to(self._device)

            if sample_data.size(0) > self.batch_size:
                shuffle_idx = torch.randperm(sample_data.size(0))
                sample_data = sample_data[shuffle_idx[:self.batch_size]]

            M_1 = torch.cat([features, sample_data], dim=0).to(self._device, non_blocking=True)
            M_2 = torch.cat([features_aug, sample_data], dim=0).to(self._device, non_blocking=True)
            M_1 = F.normalize(M_1, dim=1)
            M_2 = F.normalize(M_2, dim=1)
            sim = torch.matmul(M_1, M_2.t()) / 0.8
        else:
            features = F.normalize(features, dim=1)
            features_aug = F.normalize(features_aug, dim=1)
            sim = torch.matmul(features, features_aug.t()) / 0.8

        loss = F.cross_entropy(sim, torch.arange(sim.shape[0], device=self._device).long())
        # logging.info(f"orth_loss: {loss}")
        return loss


    def _mmda_loss(self, features:torch.Tensor):
        if not hasattr(self, '_domain_means') or len(self._domain_means) == 0:
            return 0.0

        mmda_loss = 0.0
        for domain_mean, domain_cov in zip(self._domain_means, self._domain_covs):
            domain_mean = torch.tensor(domain_mean, dtype=torch.float64).to(self._device)
            domain_cov = domain_cov.to(self._device)
            m = MultivariateNormal(domain_mean.float(), domain_cov.float())
            domain_features = m.sample(sample_shape=(features.size(0),))
            mmda_loss += self._compute_mmd(features, domain_features)

        mmda_loss /= len(self._domain_means)
        # logging.info(f"mmda_loss: {mmda_loss}")
        return mmda_loss


    def _compute_mmd(self, x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        def gaussian_kernel(source, target):
            n_samples = int(source.size()[0]) + int(target.size()[0])
            total = torch.cat([source, target], dim=0)
            total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
            total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
            L2_distance = ((total0 - total1) ** 2).sum(2) + 1e-8
            if fix_sigma:
                bandwidth = fix_sigma
            else:
                bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
            bandwidth /= kernel_mul ** (kernel_num // 2)
            bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
            kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
            return sum(kernel_val)
        
        batch_size = int(x.size()[0])
        kernels = gaussian_kernel(x, y)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        mmd = torch.mean(XX + YY - XY -YX)
        return mmd