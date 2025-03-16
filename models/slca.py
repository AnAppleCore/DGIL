import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.distributions import MultiCentroidDist
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

        self.dot_epochs = args.get('dot_epochs', 0)
        
        self.args = args
        self.seed = args['seed']
        self.task_sizes = []

        self._cur_domain = 0
        self.cls_to_task_id = {}
        self.cls_to_domain_id = {}
        self.domain_id_to_cls = {}
        self.domain_distributions = {}

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
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs, bcb_no_grad=self.fix_bcb)['logits']
                cur_targets = torch.where(targets-self._known_classes>=0, targets-self._known_classes, -100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

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

                cls_mean = torch.tensor(self._class_means_slca[c_id], dtype=torch.float64).to(self._device)
                cls_cov = self._class_covs_slca[c_id].to(self._device)
                
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
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

                ptp_inps = []
                ptp_tgts = []
                ptp_dids = []
                fake_inps = []
                fake_tgts = []
                fake_dids = []

                for d_id, cid_list in self.domain_id_to_cls.items():
                    ptp_inp = []
                    ptp_tgt = []
                    ptp_did = []
                    num_sampled_pcid = num_sampled_pcls//len(cid_list)
                    for c_id in cid_list:
                        cls_mean = torch.tensor(self._class_means_slca[c_id], dtype=torch.float64).to(self._device)
                        cls_cov = self._class_covs_slca[c_id].to(self._device)
                        m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                        ptp_inp.append(m.sample(sample_shape=(num_sampled_pcid,)))
                        ptp_tgt.extend([c_id]*num_sampled_pcid)
                        ptp_did.extend([d_id]*num_sampled_pcid)
                    ptp_inp = torch.cat(ptp_inp, dim=0).float().to(self._device)
                    ptp_tgt = torch.tensor(ptp_tgt).long().to(self._device)
                    ptp_did = torch.tensor(ptp_did).long().to(self._device)
                    ptp_inps.append(ptp_inp)
                    ptp_tgts.append(ptp_tgt)
                    ptp_dids.append(ptp_did)

                    fake_inp = self._network.domain_tsf(inp, ptp_inp)
                    fake_tgt = torch.zeros_like(tgt) + tgt
                    fake_did = torch.zeros_like(did) + d_id
                    fake_inps.append(fake_inp)
                    fake_tgts.append(fake_tgt.detach())
                    fake_dids.append(fake_did.detach())

                fake_inps = torch.cat(fake_inps, dim=0)
                fake_tgts = torch.cat(fake_tgts, dim=0)
                fake_dids = torch.cat(fake_dids, dim=0)
                ptp_inps = torch.cat(ptp_inps, dim=0)
                ptp_tgts = torch.cat(ptp_tgts, dim=0)
                ptp_dids = torch.cat(ptp_dids, dim=0)

                all_inps = torch.cat([inp, ptp_inps, fake_inps], dim=0)
                all_tgts = torch.cat([tgt, ptp_tgts, fake_tgts], dim=0)
                all_dids = torch.cat([did, ptp_dids, fake_dids], dim=0)

                all_class_outputs = self._network.class_clf(all_inps)
                cls_loss = sup_con(features=all_class_outputs, labels=all_tgts)

                all_domain_outputs = self._network.domain_clf(all_inps)
                dom_loss = sup_con(features=all_domain_outputs, labels=all_dids)

                loss = cls_loss + dom_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_cls += cls_loss.item()
                losses_dom += dom_loss.item()

                # tsne visualization
                if epoch == run_epochs-1 and _iter == crct_num-1 and self.tsne_visualize:
                    if not hasattr(self, 'img_folder'):
                        self.img_folder = f"./imgs/{time.strftime('%Y%m%d_%H%M%S')}_{self.tsne_visualize}"
                        os.makedirs(self.img_folder, exist_ok=True)
                    with torch.no_grad():
                        tsne = TSNE(n_components=2)
                        vis_features = tsne.fit_transform(all_inps.cpu().numpy())
                        cmap = plt.get_cmap('tab10')
                        
                        # paint by domain
                        domain_norm = plt.Normalize(vmin=0, vmax=self.data_manager.num_domains-1)
                        plt.figure(figsize=(8, 8))
                        plt.scatter(
                            vis_features[:len(inp), 0], vis_features[:len(inp), 1], 
                            c=cmap(domain_norm(did.cpu().numpy())), label='real', s=10, marker='s', 
                        )
                        plt.scatter(
                            vis_features[len(inp):len(inp)+len(ptp_inps), 0], vis_features[len(inp):len(inp)+len(ptp_inps), 1], 
                            c=cmap(domain_norm(ptp_dids.cpu().numpy())), label='ptp', s=10, marker='o', 
                        )
                        plt.scatter(
                            vis_features[len(inp)+len(ptp_inps):, 0], vis_features[len(inp)+len(ptp_inps):, 1], 
                            c=cmap(domain_norm(fake_dids.cpu().numpy())), label='fake', s=10, marker='^', 
                        )

                        plt.title('Task {}, Domain Transformation Epoch {} Domain-Wise'.format(self._cur_task, epoch+1))
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(f'{self.img_folder}/feat_vis_task{self._cur_task}_epoch{epoch+1}_domain.png')

                        # paint by task id
                        np_all_tgts = all_tgts.cpu().numpy()
                        np_all_tsks = np.array([self.cls_to_task_id[t] for t in np_all_tgts])
                        task_norm = plt.Normalize(vmin=0, vmax=self.data_manager.nb_tasks-1)
                        plt.figure(figsize=(8, 8))
                        plt.scatter(
                            vis_features[:len(inp), 0], vis_features[:len(inp), 1], 
                            c=cmap(task_norm(np_all_tsks[:len(inp)])), label='real', s=10, marker='s', 
                        )
                        plt.scatter(
                            vis_features[len(inp):len(inp)+len(ptp_inps), 0], vis_features[len(inp):len(inp)+len(ptp_inps), 1], 
                            c=cmap(task_norm(np_all_tsks[len(inp):len(inp)+len(ptp_inps)])), label='ptp', s=10, marker='o', 
                        )
                        plt.scatter(
                            vis_features[len(inp)+len(ptp_inps):, 0], vis_features[len(inp)+len(ptp_inps):, 1], 
                            c=cmap(task_norm(np_all_tsks[len(inp)+len(ptp_inps):])), label='fake', s=10, marker='^', 
                        )

                        plt.title('Task {}, Domain Transformation Epoch {} Task-Wise'.format(self._cur_task, epoch+1))
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(f'{self.img_folder}/feat_vis_task{self._cur_task}_epoch{epoch+1}_task.png')

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
                cls_mean = torch.tensor(self._class_means_slca[c_id], dtype=torch.float64).to(self._device)*(0.9+decay)
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

                if len(self.domain_id_to_cls.keys()) > 1 and self.dot_epochs > 0:
                    fake_inps = []
                    fake_tgts = []
                    for d_id, d_dist in self.domain_distributions.items():
                        with torch.no_grad():
                            fake_inp = self._network.domain_tsf(inp, d_dist.generate(64))
                            fake_tgt = torch.zeros_like(tgt) + tgt
                        fake_inps.append(fake_inp)
                        fake_tgts.append(fake_tgt.detach())

                    fake_inps = torch.cat(fake_inps, dim=0)
                    fake_tgts = torch.cat(fake_tgts, dim=0)
                    inp = torch.cat([inp, fake_inps], dim=0)
                    tgt = torch.cat([tgt, fake_tgts], dim=0)

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

        logging.info('Compute distributions for classes {}-{}'.format(self._known_classes, self._total_classes))

        all_vectors = np.concatenate(all_vectors, axis=0)
        all_vectors = torch.from_numpy(all_vectors).to(self._device)
        if self._cur_domain not in self.domain_distributions:
            new_domain_dist = MultiCentroidDist(
                n_centroids=32, feature_dim=self.feature_dim, device=self._device
            )
            new_domain_dist.compute_centroids(all_vectors)
            self.domain_distributions[self._cur_domain] = new_domain_dist
        else:
            old_domain_dist: MultiCentroidDist = self.domain_distributions[self._cur_domain]
            old_domain_dist.update(all_vectors)
            self.domain_distributions[self._cur_domain] = old_domain_dist

        logging.info('Compute domain distribution for domain {}'.format(self._cur_domain))



def sup_con(features, temperature=0.07, labels=None, mask=None):
    features_norm = F.normalize(features, p=2, dim=1)
    batch_size = features_norm.shape[0]
    device = features_norm.device

    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(features_norm, features_norm.T),
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    exp_logits = torch.exp(logits)

    logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
    positives_mask = mask * logits_mask
    negatives_mask = 1. - mask

    num_positives_per_row = torch.sum(positives_mask, axis=1)
    denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
        exp_logits * positives_mask, axis=1, keepdims=True)

    log_probs = logits - torch.log(denominator)
    if torch.any(torch.isnan(log_probs)):
        raise ValueError("Log_prob has nan!")

    log_probs = torch.sum(
        log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                    num_positives_per_row > 0]

    # loss
    loss = -log_probs
    loss = loss.mean()
    return loss