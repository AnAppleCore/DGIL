import logging
import os
import time
from typing import Union

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.augmentation.auto import RandAugment, TrivialAugment
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import EPSILON, BaseLearner
from utils.data_manager import DataManager
from utils.distributions import (CovarianceDist, MultiCentroidDist,
                                 MultiPrototypeDist)
from utils.domain_data_manager import DomainDataManager
from utils.inc_net import DoTPromptVitNet
from utils.toolkit import tensor2numpy


class Learner(BaseLearner):
    def __init__(self, args:dict):
        super().__init__(args)
    
        self._network = DoTPromptVitNet(args, True)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get("min_lr", 1e-8)
        self.num_workers = args.get("num_workers", 8)

        self.first_sl = args.get("first_sl", False)
        self.slow_rate = args.get("slow_rate", 0.1)

        # self.cls_con_weight = args.get("cls_con_weight", 1.00)
        # self.dom_con_weight = args.get("dom_con_weight", 1.00)
        # self.num_class_centroids = args.get("num_class_centroids", 10)

        #TODO parameter tuning
        self.ca_epochs = args.get("ca_epochs", 0)
        self.ca_lr = args.get("ca_lr", 0.001)
        self.logit_norm = args.get("logit_norm", 0.1)

        self.dot_epochs = args.get("dot_epochs", 0)
        self.dot_lr = args.get("dot_lr", 0.001)
        self.tsne_visualize = args.get("tsne_visualize", False)

        # Freeze the parameters for ViT.
        if self.args["freeze"]:
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False
        
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self._network.backbone.named_parameters():
                if n.startswith(tuple(self.args["freeze"])):
                    p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

        # distributions related
        self._cur_domain = 0
        self.class_to_task_map = {}
        self.class_to_domain_map = {}
        self.domain_to_class_map = {}
        self.task_sizes = []
        self._class_means = {}
        self.class_distributions = {}
        self.domain_distributions = {}

        # augmentation used for single-source DG
        self.rand_aug = K.AugmentationSequential(RandAugment(n=2, m=10))
        self.trivial_aug = K.AugmentationSequential(TrivialAugment())

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.restore_head()

    def incremental_train(self, data_manager: Union[DataManager, DomainDataManager] = None):
        """
        The basic incremental learning training process.
        """
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        try:
            self._cur_domain = data_manager.get_cur_domain(self._cur_task)
        except:
            self._cur_domain = 0
        self._total_classes = self._known_classes + task_size
        self._network.update_head(task_size)
        for cls in range(self._known_classes, self._total_classes):
            self.class_to_task_map[cls] = self._cur_task
            self.class_to_domain_map[cls] = self._cur_domain
            if self._cur_domain not in self.domain_to_class_map:
                self.domain_to_class_map[self._cur_domain] = []
            self.domain_to_class_map[self._cur_domain].append(cls)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.data_manager = data_manager
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        if len(self._multiple_gpus) > 1:
            logging.info('Using Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader) # prompt tuning
        self._compute_distributions(self.train_loader) # compute class and domain distributions
        if len(self.domain_to_class_map.keys())>1 and self.dot_epochs > 0:
            self._train_domain_transformation()
        if self._cur_task > 0 and self.ca_epochs > 0:
            self._train_classifier() # rectify head by pseudo instructed features
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
            
        if self._cur_task > 0:
            self._init_prompt(optimizer)

        if self._cur_task > 0 and self.args["reinit_optimizer"]:
            optimizer = self.get_optimizer()

        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            if len(self._multiple_gpus) > 1:
                self._network.module.backbone.train()
                self._network.module.original_backbone.eval()
            else:
                self._network.backbone.train()
                self._network.original_backbone.eval()

            losses = 0.0
            orth_losses, mmda_losses = 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                output = self._network(inputs, task_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')
                features = output["pre_logits"]

                loss = F.cross_entropy(logits, targets.long())
                if self.args["pull_constraint"] and 'reduce_sim' in output:
                    loss = loss - self.args["pull_constraint_coeff"] * output['reduce_sim']

                # inputs_aug = self.trivial_aug(inputs)
                # outputs_aug = self._network(inputs_aug, task_id=self._cur_task, train=True)
                # logits_aug = outputs_aug["logits"][:, :self._total_classes]
                # logits_aug[:, :self._known_classes] = float('-inf')
                # features_aug = outputs_aug["pre_logits"]

                # loss += F.cross_entropy(logits_aug, targets.long())
                # if self.args["pull_constraint"] and 'reduce_sim' in outputs_aug:
                #     loss -= self.args["pull_constraint_coeff"] * outputs_aug['reduce_sim']

                # if self.cls_con_weight > 0:
                #     orth_loss = self.cls_con_weight * self.orth_loss(features, features_aug)
                #     loss += orth_loss
                #     orth_losses += orth_loss.item() if orth_loss != 0.0 else 0.0

                # if self.dom_con_weight > 0:
                #     mmda_loss = self.dom_con_weight * self.mmda_loss(torch.cat([features, features_aug], dim=0))
                #     loss += mmda_loss
                #     mmda_losses += mmda_loss.item() if mmda_loss != 0.0 else 0.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f} (orth {:.3f}, mmda {:.3f}), Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    orth_losses / len(train_loader),
                    mmda_losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f} (orth {:.3f}, mmda {:.3f}), Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    orth_losses / len(train_loader),
                    mmda_losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

            logging.info(info)

    def _compute_distributions(self, data_loader):
        if len(self._multiple_gpus) > 1:
            self._network.module.backbone.eval()
            self._network.module.original_backbone.eval()
        else:
            self._network.backbone.eval()
            self._network.original_backbone.eval()

        # for class distribution
        features = []
        raw_features = []
        labels = []
        for i, (_, inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            targets = torch.index_select(targets, 0, mask)
            with torch.no_grad():
                output = self._network(inputs, task_id=self._cur_task, train=True)
                feature = output["pre_logits"]
                raw_feature = output["raw_features"]
            features.append(feature)
            raw_features.append(raw_feature)
            labels.append(targets)
        features = torch.cat(features, dim=0)
        raw_features = torch.cat(raw_features, dim=0)
        labels = torch.cat(labels, dim=0)

        unique_labels: torch.Tensor = torch.unique(labels)
        for cls_label in unique_labels:
            mask = (labels == cls_label).nonzero().view(-1)
            cls_feature = features[mask]
            cls_raw_feature = raw_features[mask]
            cls_label_id = int(cls_label)
            if cls_label_id in self.class_distributions:
                # error since the class set is disjoint across tasks
                raise ValueError(f"Class set is disjoint across tasks: repeated class {cls_label_id}")
            else:
                # new_distribution = MultiCentroidDist(n_centroids=self.num_class_centroids, feature_dim=cls_feature.shape[-1], device=self._device)
                # new_distribution.compute_centroids(cls_feature)
                # self.class_distributions[cls_label_id] = new_distribution
                # self._class_means[cls_label_id] = cls_feature.mean(dim=0)

                mean = cls_feature.mean(dim=0)
                cov = torch.cov(cls_feature.t())
                new_distribution = CovarianceDist(feature_dim=cls_feature.shape[-1], device=self._device)
                new_distribution.init_from(mean, cov, len(cls_feature))
                self.class_distributions[cls_label_id] = new_distribution
                self._class_means[cls_label_id] = mean
                
                # closest_indices = new_distribution.closest_id(cls_feature)
                # cluster_indices = new_distribution.cluster_masks

                # # TODO store covariance matrix instead of variance?
                # new_ins_distribution = MultiPrototypeDist(n_prototypes=self.num_class_centroids, feature_dim=cls_feature.shape[-1], device=self._device)
                # new_ins_distribution.init_from(closest_indices, cluster_indices, cls_feature)
                # self.ins_cls_proto_dists[cls_label_id] = new_ins_distribution

                # new_uni_distribution = MultiPrototypeDist(n_prototypes=self.num_class_centroids, feature_dim=cls_raw_feature.shape[-1], device=self._device)
                # new_uni_distribution.init_from(closest_indices, cluster_indices, cls_raw_feature)
                # self.uni_cls_proto_dists[cls_label_id] = new_uni_distribution
        logging.info(f"Distributions computed for {unique_labels.shape[0]} classes: {unique_labels.tolist()} ")

        # for domain distribution
        # if self.dom_con_weight > 0:
        mean = features.mean(dim=0)
        cov = torch.cov(features.t())
        if self._cur_domain in self.domain_distributions:
            # update the existing domain distribution
            existing_distribution: CovarianceDist = self.domain_distributions[self._cur_domain]
            existing_distribution.update(mean, cov, len(features))
            self.domain_distributions[self._cur_domain] = existing_distribution
        else:
            new_distribution = CovarianceDist(feature_dim=features.shape[-1], device=self._device)
            new_distribution.init_from(mean, cov, len(features))
            self.domain_distributions[self._cur_domain] = new_distribution
        logging.info(f"Distributions computed for domain {self._cur_domain}")

    def _train_domain_transformation(self):
        run_epochs = self.dot_epochs
        crct_num = self._total_classes
        self._network.reset_domain_tsf_clf(
            cls_feat_dim=512, dom_feat_dim=512
        )
        param_list = {}
        for n, p in self._network.named_parameters():
            if "domain" in n or "class" in n:
                p.requires_grad = True
                param_list[n] = p
        network_params = [{'params': param_list.values(), 'lr': self.dot_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.dot_lr, momentum=0.9, weight_decay=self.weight_decay)
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
                t_id = self.class_to_task_map[c_id]
                d_id = self.class_to_domain_map[c_id]

                m: CovarianceDist = self.class_distributions[c_id]

                sampled_data_single = m.generate(num_sampled_pcls)
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

                for d_id, cid_list in self.domain_to_class_map.items():
                    ptp_inp = []
                    ptp_tgt = []
                    ptp_did = []
                    num_sampled_pcid = num_sampled_pcls//len(cid_list)
                    for c_id in cid_list:
                        m: CovarianceDist = self.class_distributions[c_id]
                        ptp_inp.append(m.generate(num_sampled_pcid))
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
                    self.img_folder = f"./imgs/{time.strftime('%Y%m%d_%H%M%S')}" if not hasattr(self, 'img_folder') else self.img_folder
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
                        np_all_tsks = np.array([self.class_to_task_map[t] for t in np_all_tgts])
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

    def _train_classifier(self):
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.back_up_head()
        for p in self._network.rec_head.parameters():
            p.requires_grad = True
        param_list = [p for p in self._network.rec_head.parameters() if p.requires_grad]
        param_groups = [{'params': param_list, 'lr': self.ca_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(param_groups, lr=self.ca_lr, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.ca_epochs, eta_min=self.min_lr)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)
        self._network.eval()

        prog_bar = tqdm(range(self.ca_epochs), desc="Head training")
        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256

            for c_id in range(self._total_classes):
                t_id = self.class_to_task_map[c_id]
                decay = (t_id + 1) / (self._cur_task + 1) * 0.1

                m: CovarianceDist = self.class_distributions[c_id]
                sampled_data_single = m.generate(num_sampled_pcls, decay)
                sampled_data.append(sampled_data_single)
                sampled_label.append(
                    torch.ones(num_sampled_pcls, device=self._device).long() * c_id
                )

            inputs = torch.cat(sampled_data, dim=0).float().to(self._device)
            label = torch.cat(sampled_label, dim=0).long().to(self._device)

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            label = label[sf_indexes]

            for _iter in range(self._total_classes):
                inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                tgt = label[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]

                if len(self.domain_to_class_map.keys()) > 1 and self.dot_epochs > 0:
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

                outputs = self._network(inp, head_only=True)
                logits = outputs['logits']

                logits = self.logit_normalize(logits)
                loss = F.cross_entropy(logits, tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

            scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = "Head Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.ca_epochs,
                losses / self._total_classes,
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
        logging.info(info)

    def get_optimizer(self):

        if self.first_sl and self._cur_task == 0:
            prompt_lrate = self.init_lr * self.slow_rate
        else:
            prompt_lrate = self.init_lr

        prompt_params, output_head_params, other_params = [], [], []
        for name, param in self._network.named_parameters():
            if param.requires_grad and 'prompt' in name:
                prompt_params.append(param)
                # logging.info(f"Prompt parameter: {name} with lr {prompt_lrate}")
            elif param.requires_grad and 'head' in name:
                output_head_params.append(param)
                # logging.info(f"Output head parameter: {name}")
            elif param.requires_grad:
                other_params.append(param)
                # logging.info(f"Other parameter: {name}")

        param_groups = [
            {'params': prompt_params, 'lr': prompt_lrate},
            {'params': output_head_params, 'lr': self.init_lr},
            {'params': other_params, 'lr': self.init_lr}
        ]

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                param_groups,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                param_groups, 
                weight_decay=self.weight_decay
            )

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_prompt(self, optimizer):
        args = self.args
        model = self._network.backbone
        task_id = self._cur_task

        # Transfer previous learned prompt params to the new prompt
        if args["prompt_pool"] and args["shared_prompt_pool"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

                with torch.no_grad():
                    model.prompt.prompt.grad.zero_()
                    model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                    optimizer.param_groups[0]['params'] = model.parameters()
                
        # Transfer previous learned prompt param keys to the new prompt
        if args["prompt_pool"] and args["shared_prompt_key"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

            with torch.no_grad():
                model.prompt.prompt_key.grad.zero_()
                model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                optimizer.param_groups[0]['params'] = model.parameters()

    def _eval_cnn(self, loader):
        self._network.eval()
        # TODO use multi-domain instructed/uninstructed feature for cnn?
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def _eval_nme(self, loader, class_means:dict):
        self._network.eval()
        # TODO use multi-domain instructed/uninstructed feature for nme?
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        class_means_np = torch.cat([class_means[i].view(1, -1) for i in range(self._total_classes)], dim=0)
        class_means_np = tensor2numpy(class_means_np)
        class_means_np = (class_means_np.T / (np.linalg.norm(class_means_np.T, axis=0) + EPSILON)).T
        
        dists = cdist(class_means_np, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance
        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        # here the extracted vectors are the instructed features
        self._network.eval()
        self._network.original_backbone.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(_inputs.to(self._device), task_id=self._cur_task)
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.extract_vector(_inputs.to(self._device), task_id=self._cur_task)
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _compute_accuracy(self, model, loader):
        model.eval()
        # TODO use multi-domain instructed/uninstructed feature for acc computation?
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    def orth_loss(self, features:torch.Tensor, features_aug:torch.Tensor):
        """
        Computes the orthogonality loss using contrastive learning.

        Args:
            features (torch.Tensor): Feature embeddings of shape (batch_size, embedding_dim).
            targets (torch.Tensor): Class labels of shape (batch_size,).

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        if self.class_distributions:
            sample_mean = []
            batch_size = features.shape[0]
            num_samples_per_class = batch_size // self._known_classes + 1
            for class_id, class_dist in self.class_distributions.items():
                if isinstance(class_dist, MultiCentroidDist):
                    sample_mean.append(class_dist.generate(num_samples_per_class))
                elif isinstance(class_dist, CovarianceDist):
                    sample_mean.append(class_dist.generate(num_samples_per_class))
            sample_mean = torch.cat(sample_mean, dim=0).to(self._device, non_blocking=True)
            if sample_mean.shape[0] > batch_size:
                shuffle_idx = torch.randperm(sample_mean.shape[0])[:batch_size]
                sample_mean = sample_mean[shuffle_idx]

            M_1 = torch.cat([features, sample_mean], dim=0).to(self._device, non_blocking=True)
            M_2 = torch.cat([features_aug, sample_mean], dim=0).to(self._device, non_blocking=True)
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

    def mmda_loss(self, features:torch.Tensor):
        if not hasattr(self, 'domain_distributions') or len(self.domain_distributions.keys()) == 0:
            return 0.0

        mmda_loss = 0.0
        for domain_id, domain_dist in self.domain_distributions.items():
            if isinstance(domain_dist, MultiCentroidDist):
                domain_features = domain_dist.generate(features.shape[0])
            elif isinstance(domain_dist, CovarianceDist):
                domain_features = domain_dist.generate(features.shape[0])
            mmda_loss += self._compute_mmd(features, domain_features)

        mmda_loss /= len(self.domain_distributions.keys())
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

    def div_loss(self, features: torch.Tensor, features_aug: torch.Tensor, targets: torch.Tensor):
        """
        moment matching loss for the domain features

        Args:
            features (torch.Tensor): Feature embeddings of shape (batch_size, embedding_dim).
            features_aug (torch.Tensor): Augmented feature embeddings of shape (batch_size, embedding_dim).
            targets (torch.Tensor): Class labels of shape (batch_size,).

        Returns:
            torch.Tensor: The computed moment matching loss.
        """

        features = F.normalize(features, dim=1, p=2, eps=EPSILON)
        features_aug = F.normalize(features_aug, dim=1, p=2, eps=EPSILON)

        sim_1 = torch.matmul(features, features_aug.detach().t()) / 0.1
        sim_2 = torch.matmul(features_aug, features.detach().t()) / 0.1

        loss = (F.cross_entropy(sim_1, torch.arange(sim_1.shape[0], device=self._device).long()) + \
                F.cross_entropy(sim_2, torch.arange(sim_2.shape[0], device=self._device).long())) / 2

        # logging.info(f"div_ce_loss: {loss}")

        if self._cur_task > 0:
            sample_mean = []
            batch_size = features.shape[0]
            num_samples_per_domain = batch_size // len(self.domain_distributions) + 1
            for domain_id, domain_dist in self.domain_distributions.items():
                if isinstance(domain_dist, MultiCentroidDist):
                    sample_mean.append(domain_dist.get_means_vector())
                elif isinstance(domain_dist, CovarianceDist):
                    sample_mean.append(domain_dist.generate(num_samples_per_domain))
            sample_mean = torch.cat(sample_mean, dim=0).to(self._device, non_blocking=True)
            if sample_mean.shape[0] > batch_size:
                shuffle_idx = torch.randperm(sample_mean.shape[0])[:batch_size]
                sample_mean = sample_mean[shuffle_idx]

            sample_mean = F.normalize(sample_mean, dim=1, p=2, eps=EPSILON)

            mse_loss = (torch.dist(features, sample_mean.detach(), p=2)+ \
                        torch.dist(features_aug, sample_mean.detach(), p=2)) / 2

            # logging.info(f"div_mse_loss: {mse_loss}")
            loss += mse_loss * 0.1

        # logging.info(f"div_loss: {loss}")
        return loss

    def logit_normalize(self, logits: torch.Tensor):
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
                
            norms_all = torch.norm(logits[:, :self._total_classes], p=2, dim=-1, keepdim=True) + 1e-7
            logits = torch.div(logits[:, :self._total_classes], norms) / self.logit_norm
        else:
            logits = logits[:, :self._total_classes]
        return logits


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