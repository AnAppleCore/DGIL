import logging
from typing import List, Union

import kornia.augmentation as K
import numpy as np
import torch
from kornia.augmentation.auto import RandAugment, TrivialAugment
from scipy.spatial.distance import cdist
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import EPSILON, BaseLearner
from utils.data_manager import DataManager
from utils.distributions import *
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

        self.contrastive_temp = args.get("contrastive_temp", 0.8)
        self.cls_con_weight = args.get("cls_con_weight", 1.00)
        self.dom_con_weight = args.get("dom_con_weight", 1.00)
        self.num_class_centroids = args.get("num_class_centroids", 10)
        self.use_multicentroid_nme = args.get("use_multicentroid_nme", False)

        #TODO parameter tuning
        self.ca_epochs = args.get("ca_epochs", 0)
        self.ca_lr = args.get("ca_lr", 0.001)
        self.logit_norm = args.get("logit_norm", 0.1)
        self.fake_ce_loss_weight = args.get("fake_ce_loss_weight", 1.0)

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
        self.task_sizes = []
        self._class_means = {}
        self.class_distributions = {}
        self.ins_domain_dists = {}
        self.uni_domain_dists = {}
        self.ins_cls_dists = {}
        self.uni_cls_dists = {}

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
        if self.ca_epochs > 0:
            self._train_classifier(self.test_loader) # rectify head by pseudo instructed features
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

                # class contrastive loss
                loss += self.cls_con_weight * self.orth_loss(features, targets)

                # domain contrastive loss
                inputs_aug = self.trivial_aug(inputs)
                # inputs_aug = self.rand_aug(inputs)
                features_aug = self._network(inputs_aug, task_id=self._cur_task, train=True)["pre_logits"]
                loss += self.dom_con_weight * self.div_loss(features, features_aug, targets)

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
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
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
                # TODO merge class distribution with instructed class distribution ?
                new_distribution = MultiCentroidDist(n_centroids=self.num_class_centroids, feature_dim=cls_feature.shape[-1], device=self._device)
                new_distribution.compute_centroids(cls_feature)
                self.class_distributions[cls_label_id] = new_distribution
                if self.use_multicentroid_nme:
                    self._class_means[cls_label_id] = new_distribution.cluster_means
                else:
                    self._class_means[cls_label_id] = cls_feature.mean(dim=0)

                #### Covariance version of class distribution
                # mean = cls_feature.mean(dim=0)
                # cov = torch.cov(cls_feature.t())
                # new_distribution = CovarianceDist(feature_dim=cls_feature.shape[-1], device=self._device)
                # new_distribution.init_from(mean, cov, len(cls_feature))
                # self.class_distributions[cls_label_id] = new_distribution
                # if self.use_multicentroid_nme:
                #     raise NotImplementedError("Multicentroid NME not implemented for CovarianceDist")
                # else:
                #     self._class_means[cls_label_id] = mean
                

                #### For instructed and uninsructed feature distributions
                # closest_indices = new_distribution.closest_id(cls_feature)
                # cluster_indices = new_distribution.cluster_masks

                # new_ins_distribution = MultiPrototypeDist(n_prototypes=self.num_class_centroids, feature_dim=cls_feature.shape[-1], device=self._device)
                # new_ins_distribution.init_from(closest_indices, cluster_indices, cls_feature)
                # self.ins_cls_proto_dists[cls_label_id] = new_ins_distribution

                # new_uni_distribution = MultiPrototypeDist(n_prototypes=self.num_class_centroids, feature_dim=cls_raw_feature.shape[-1], device=self._device)
                # new_uni_distribution.init_from(closest_indices, cluster_indices, cls_raw_feature)
                # self.uni_cls_proto_dists[cls_label_id] = new_uni_distribution

                ins_mean = cls_feature.mean(dim=0)
                ins_cov = torch.cov(cls_feature.t())
                new_ins_distribution = CovarianceDist(feature_dim=cls_feature.shape[-1], device=self._device)
                new_ins_distribution.init_from(ins_mean, ins_cov, len(cls_feature))
                self.ins_cls_dists[cls_label_id] = new_ins_distribution

                uni_mean = cls_raw_feature.mean(dim=0)
                uni_cov = torch.cov(cls_raw_feature.t())
                new_uni_distribution = CovarianceDist(feature_dim=cls_raw_feature.shape[-1], device=self._device)
                new_uni_distribution.init_from(uni_mean, uni_cov, len(cls_raw_feature))
                self.uni_cls_dists[cls_label_id] = new_uni_distribution

        logging.info(f"Distributions computed for {unique_labels.shape[0]} classes: {unique_labels.tolist()} ")

        # for domain distribution
        ins_mean = features.mean(dim=0)
        ins_cov = torch.cov(features.t())
        uni_mean = raw_features.mean(dim=0)
        uni_cov = torch.cov(raw_features.t())

        if self._cur_domain in self.ins_domain_dists:
            # update the existing domain distribution
            existing_distribution: CovarianceDist = self.ins_domain_dists[self._cur_domain]
            existing_distribution.update(ins_mean, ins_cov, len(features))
            self.ins_domain_dists[self._cur_domain] = existing_distribution

            existing_distribution: CovarianceDist = self.uni_domain_dists[self._cur_domain]
            existing_distribution.update(uni_mean, uni_cov, len(raw_features))
            self.uni_domain_dists[self._cur_domain] = existing_distribution
        else:
            new_distribution = CovarianceDist(feature_dim=features.shape[-1], device=self._device)
            new_distribution.init_from(ins_mean, ins_cov, len(features))
            self.ins_domain_dists[self._cur_domain] = new_distribution

            new_distribution = CovarianceDist(feature_dim=raw_features.shape[-1], device=self._device)
            new_distribution.init_from(uni_mean, uni_cov, len(raw_features))
            self.uni_domain_dists[self._cur_domain] = new_distribution

        logging.info(f"Distributions computed for domain {self._cur_domain}")

        ins_dist: CovarianceDist = self.ins_domain_dists[self._cur_domain]
        uni_dist: CovarianceDist = self.uni_domain_dists[self._cur_domain]
        if len(self._multiple_gpus) > 1:
            self._network.module.update_domain_transform(self._cur_domain, ins_dist, uni_dist)
        else:
            self._network.update_domain_transform(self._cur_domain, ins_dist, uni_dist)

        logging.info(f"Domain transform updated for domain {self._cur_domain}")

    def _train_classifier(self, test_loader):
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

            correct_fake, correct_fake_2 = 0, 0
            total_fake, total_fake_2 = 0, 0

            sampled_ins_feature = []
            sampled_uni_feature = []
            sampled_label = []
            sampled_domain_id = []
            num_sampled_pcls = self.batch_size * 5

            for c_id in range(self._total_classes):
                t_id = self.class_to_task_map[c_id]
                decay = (t_id + 1) / (self._cur_task + 1) * 0.1

                ins_dist: CovarianceDist = self.ins_cls_dists[c_id]
                uni_dist: CovarianceDist = self.uni_cls_dists[c_id]
                ins_feature = ins_dist.generate(num_sampled_pcls, decay=decay)
                uni_feature = uni_dist.generate(num_sampled_pcls, decay=decay)

                sampled_ins_feature.append(ins_feature)
                sampled_uni_feature.append(uni_feature)
                sampled_label.append(
                    torch.ones(num_sampled_pcls, device=self._device).long() * c_id
                )
                sampled_domain_id.append(
                    torch.ones(num_sampled_pcls, device=self._device).long() * self.class_to_domain_map[c_id]
                )

            ins_feature = torch.cat(sampled_ins_feature, dim=0).float().to(self._device)
            uni_feature = torch.cat(sampled_uni_feature, dim=0).float().to(self._device)
            label = torch.cat(sampled_label, dim=0).long().to(self._device)
            domain_id = torch.cat(sampled_domain_id, dim=0).long().to(self._device)

            sf_indexes = torch.randperm(ins_feature.size(0))
            ins_feature = ins_feature[sf_indexes]
            uni_feature = uni_feature[sf_indexes]
            label = label[sf_indexes]
            domain_id = domain_id[sf_indexes]

            for _iter in range(self._total_classes):
                ins_inp = ins_feature[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                uni_inp = uni_feature[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                tgt = label[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                dom_id = domain_id[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]

                # generate fake ins features
                with torch.no_grad():
                    if isinstance(self._network, nn.DataParallel):
                        fake_ins = self._network.module.uni_to_ins(uni_inp, domain_id=dom_id)
                    else:
                        fake_ins = self._network.uni_to_ins(uni_inp, domain_id=dom_id)

                    fake_uni_ls = []
                    fake_uni_2_ls = []

                    for d_id in range(len(self.ins_domain_dists)):
                        d_id_tensor = torch.ones(num_sampled_pcls, device=self._device).long() * d_id
                        if isinstance(self._network, nn.DataParallel):
                            fake_uni = self._network.module.ins_to_uni(ins_inp, domain_id=d_id_tensor)
                            fake_uni_2 = self._network.module.ins_to_uni(fake_ins, domain_id=d_id_tensor)
                        else:
                            fake_uni = self._network.ins_to_uni(ins_inp, domain_id=d_id_tensor)
                            fake_uni_2 = self._network.ins_to_uni(fake_ins, domain_id=d_id_tensor)
                        fake_uni_ls.append(fake_uni)
                        fake_uni_2_ls.append(fake_uni_2)
                
                fake_uni_ls = torch.cat(fake_uni_ls, dim=0).detach()
                fake_uni_2_ls = torch.cat(fake_uni_2_ls, dim=0).detach()

                logit_uni = self._network(uni_inp, head_only=True)['logits']
                logit_fake_uni = self._network(fake_uni_ls, head_only=True)['logits']
                logit_fake_uni_2 = self._network(fake_uni_2_ls, head_only=True)['logits']

                logit_uni = self.logit_normalize(logit_uni)
                logit_fake_uni = self.logit_normalize(logit_fake_uni)
                logit_fake_uni_2 = self.logit_normalize(logit_fake_uni_2)

                loss_uni = F.cross_entropy(logit_uni, tgt)

                tgt_for_fake = tgt.repeat(fake_uni_ls.size(0)//tgt.size(0))
                loss_fake_uni = F.cross_entropy(logit_fake_uni, tgt_for_fake)
                loss_fake_uni_2 = F.cross_entropy(logit_fake_uni_2, tgt_for_fake)

                loss = loss_uni + (loss_fake_uni + loss_fake_uni_2) * self.fake_ce_loss_weight

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logit_uni, dim=1)
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                _, preds = torch.max(logit_fake_uni, dim=1)
                correct_fake += preds.eq(tgt_for_fake.expand_as(preds)).cpu().sum()
                total_fake += len(tgt_for_fake)

                _, preds = torch.max(logit_fake_uni_2, dim=1)
                correct_fake_2 += preds.eq(tgt_for_fake.expand_as(preds)).cpu().sum()
                total_fake_2 += len(tgt_for_fake)

            scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            train_acc_fake = np.around(tensor2numpy(correct_fake) * 100 / total_fake, decimals=2)
            train_acc_fake_2 = np.around(tensor2numpy(correct_fake_2) * 100 / total_fake_2, decimals=2)

            if (epoch + 1) % 5 == 0 or epoch == self.ca_epochs - 1:
                test_acc = self._compute_accuracy(self._network, test_loader, use_uninstructed=True)
                info = "Head Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Train_accy_fake {:.2f}, Train_accy_fake_2 {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.ca_epochs,
                    losses / len(ins_feature),
                    train_acc,
                    train_acc_fake,
                    train_acc_fake_2,
                    test_acc,
                )
            else:
                info = "Head Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Train_accy_fake {:.2f}, Train_accy_fake_2 {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.ca_epochs,
                    losses / len(ins_feature),
                    train_acc,
                    train_acc_fake,
                    train_acc_fake_2,
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
        # TODO use multi-domain instructed feature for cnn?
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, task_id=self._cur_task, use_uninstructed=True)["logits"][:, :self._total_classes]
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
        # TODO use multi-domain instructed feature for nme?
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        if self.use_multicentroid_nme:
            scores_list = []
            for centroid_id in range(self.num_class_centroids):
                class_means_np = torch.cat(
                    [class_means[i][centroid_id].view(1, -1) for i in range(self._total_classes)], dim=0
                )
                class_means_np = tensor2numpy(class_means_np)
                class_means_np = (class_means_np.T / (np.linalg.norm(class_means_np.T, axis=0) + EPSILON)).T

                dists = cdist(class_means_np, vectors, "sqeuclidean")  # [nb_classes, N]
                scores = dists.T  # [N, nb_classes], choose the one with the smallest distance
                scores_list.append(scores)

            scores_list = np.stack(scores_list, axis=1)  # [N, num_class_centroids, nb_classes]
            average_scores = np.mean(scores_list, axis=1)  # [N, nb_classes]
            return np.argsort(average_scores, axis=1)[:, : self.topk], y_true  # [N, topk]

        else:
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

    def _compute_accuracy(self, model, loader, use_uninstructed=False):
        model.eval()
        # TODO use multi-domain instructed feature for acc computation?
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, task_id=self._cur_task, use_uninstructed=use_uninstructed)["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    def orth_loss(self, features:torch.Tensor, targets:torch.Tensor):
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
            num_samples_per_class = batch_size // self._total_classes + 1
            for class_id, class_dist in self.class_distributions.items():
                if isinstance(class_dist, MultiCentroidDist):
                    sample_mean.append(class_dist.get_means_vector())
                elif isinstance(class_dist, CovarianceDist):
                    sample_mean.append(class_dist.generate(num_samples_per_class))
            sample_mean = torch.cat(sample_mean, dim=0).to(self._device, non_blocking=True)
            if sample_mean.shape[0] > batch_size:
                shuffle_idx = torch.randperm(sample_mean.shape[0])[:batch_size]
                sample_mean = sample_mean[shuffle_idx]

            M = torch.cat([sample_mean, features], dim=0).to(self._device, non_blocking=True)
            M = F.normalize(M, dim=1)
            sim = torch.matmul(M, M.t()) / self.contrastive_temp

        else:
            features = F.normalize(features, dim=1)
            sim = torch.matmul(features, features.t()) / self.contrastive_temp

        loss = F.cross_entropy(sim, torch.arange(sim.shape[0], device=self._device).long())
        # logging.info(f"orth_loss: {loss}")
        return loss

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
            num_samples_per_domain = batch_size // len(self.ins_domain_dists) + 1
            for domain_id, domain_dist in self.ins_domain_dists.items():
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