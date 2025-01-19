import logging
from typing import Union

import numpy as np
import torch
from models.base import EPSILON, BaseLearner
from scipy.spatial.distance import cdist
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
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

        self.use_cls_con_loss = args.get("use_cls_con_loss", False)
        self.use_dom_con_loss = args.get("use_dom_con_loss", False)
        self.contrastive_temp = args.get("contrastive_temp", 0.8)
        self.cls_con_weight = args.get("cls_con_weight", 0.01)
        self.dom_con_weight = args.get("dom_con_weight", 0.01)
        self.num_class_centroids = args.get("num_class_centroids", 10)

        self.use_multicentroid_nme = args.get("use_multicentroid_nme", False)

        # Freeze the parameters for ViT.
        if self.args["freeze"]:
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False
        
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self._network.backbone.named_parameters():
                if n.startswith(tuple(self.args["freeze"])):
                    p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

        # distributions related
        self._cur_domain = 0
        self._class_means = {}
        self.domain_distributions = {}
        self.class_distributions = {}
        self.raw_class_distributions = {}

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager: Union[DataManager, DomainDataManager] = None):
        """
        The basic incremental learning training process.
        """
        self._cur_task += 1
        try:
            self._cur_domain = data_manager.get_cur_domain(self._cur_task)
        except:
            self._cur_domain = 0
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
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
        self._train(self.train_loader, self.test_loader)
        self._compute_distributions(self.train_loader)
        # self._train_dot_and_doh(self.train_loader, self.test_loader)
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
            self._network.backbone.train()
            self._network.original_backbone.eval()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                output = self._network(inputs, task_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                loss = F.cross_entropy(logits, targets.long())
                if self.args["pull_constraint"] and 'reduce_sim' in output:
                    loss = loss - self.args["pull_constraint_coeff"] * output['reduce_sim']

                if self.use_cls_con_loss:
                    features = output["pre_logits"]
                    loss += self.cls_con_weight * self.orth_loss(features, targets)

                if self.use_dom_con_loss:
                    shuffled_output = self._network(inputs, task_id=self._cur_task, train=True, shuffle_tokens=True)
                    shuffled_features = shuffled_output["pre_logits"]
                    domain_ids = torch.zeros(len(inputs), dtype=torch.long, device=self._device) + self._cur_domain
                    loss += self.dom_con_weight * self.div_loss(shuffled_features, domain_ids)

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
        logging.info("Computing distributions...")
        self._network.backbone.eval()
        self._network.original_backbone.eval()
        # for domain distribution
        if self.use_dom_con_loss:
            features, labels = [], []
            for i, (_, inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)
                with torch.no_grad():
                    feature = self._network(inputs, task_id=self._cur_task, train=True, shuffle_tokens=True)["pre_logits"]
                features.append(feature)
                labels.append(targets)
            features = torch.cat(features, dim=0)
            labels = torch.cat(labels, dim=0)

            mean = features.mean(dim=0)
            cov = torch.mm(features.t(), features) / len(features)
            if self._cur_domain in self.domain_distributions:
                # update the existing domain distribution
                existing_distribution: CovarianceDist = self.domain_distributions[self._cur_domain]
                existing_distribution.update(mean, cov, len(features))
                self.domain_distributions[self._cur_domain] = existing_distribution
            else:
                new_distribution = CovarianceDist(feature_dim=features.shape[-1], device=self._device)
                new_distribution.update(mean, cov, len(features))
                self.domain_distributions[self._cur_domain] = new_distribution
            logging.info(f"Distributions computed for domain {self._cur_domain}")

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
                feature = self._network(inputs, task_id=self._cur_task, train=True)["pre_logits"]
                raw_feature = self._network.forward_uninstructed_features(inputs)
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
                new_distribution = MultiCentroidDist(n_centroids=self.num_class_centroids, feature_dim=cls_feature.shape[-1], device=self._device)
                new_distribution.compute_centroids(cls_feature)
                self.class_distributions[cls_label_id] = new_distribution
                if self.use_multicentroid_nme:
                    self._class_means[cls_label_id] = new_distribution.cluster_means
                else:
                    self._class_means[cls_label_id] = cls_feature.mean(dim=0) # new_distribution.get_means_vector()
                
                new_raw_distribution = MultiPrototypeDist(n_prototypes=self.num_class_centroids, feature_dim=cls_raw_feature.shape[-1], device=self._device)
                closest_indices = new_distribution.closest_id(cls_feature)
                new_raw_distribution.update(closest_indices, cls_raw_feature)
                self.raw_class_distributions[cls_label_id] = new_raw_distribution
        logging.info(f"Distributions computed for {unique_labels.shape[0]} classes: {unique_labels.tolist()} ")

    def _train_dot_and_doh(self, train_loader, test_loader):
        self._network.to(self._device)

        # freeze and record the parameters except the domain transformation mlp and domain-specific head
        active_params_dict = {}
        for name, param in self._network.named_parameters():
            if param.requires_grad and 'domain' not in name:
                active_params_dict[name] = param
                param.requires_grad = False

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        prog_bar = tqdm(range(self.args['tuned_epoch']), desc="DoT & DoH")
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            self._network.original_backbone.eval()

            losses = 0.0
            correct, total = 0, 0
        pass

    def get_optimizer(self):

        if self.first_sl and self._cur_task == 0:
            prompt_lrate = self.init_lr * self.slow_rate
        else:
            prompt_lrate = self.init_lr

        prompt_params, output_head_params, other_params = [], [], []
        for name, param in self._network.named_parameters():
            if param.requires_grad and 'prompt' in name:
                prompt_params.append(param)
                logging.info(f"Prompt parameter: {name} with lr {prompt_lrate}")
            elif param.requires_grad and 'head' in name:
                output_head_params.append(param)
                logging.info(f"Output head parameter: {name}")
            elif param.requires_grad:
                other_params.append(param)
                logging.info(f"Other parameter: {name}")

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
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
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
            sample_targets = []
            batch_size = features.shape[0]
            sample_per_class = batch_size // self._known_classes + 1
            for class_id, class_dist in self.class_distributions.items():
                if isinstance(class_dist, MultiCentroidDist):
                    sample_mean.append(class_dist.generate(sample_per_class))
                    sample_targets.append(torch.ones(sample_per_class, dtype=torch.long) * class_id)
            sample_mean = torch.cat(sample_mean, dim=0).to(self._device, non_blocking=True)
            sample_targets = torch.cat(sample_targets, dim=0).to(self._device, non_blocking=True)
            if sample_mean.shape[0] > batch_size:
                shuffle_idx = torch.randperm(sample_mean.shape[0])[:batch_size]
                sample_mean = sample_mean[shuffle_idx]
                sample_targets = sample_targets[shuffle_idx]

            M = torch.cat([sample_mean, features], dim=0).to(self._device, non_blocking=True)
            T = torch.cat([sample_targets, targets], dim=0).to(self._device, non_blocking=True)
            M = F.normalize(M, dim=1)
            sim = torch.matmul(M, M.t()) / self.contrastive_temp
            mask = (T.unsqueeze(0) == T.unsqueeze(1)).float()

        else:
            features = F.normalize(features, dim=1)
            sim = torch.matmul(features, features.t()) / self.contrastive_temp
            mask = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()

        mask.fill_diagonal_(0)
        positive_sim = sim[mask == 1]
        if positive_sim.numel() > 0:
            max_positive = positive_sim.max()
            positive_loss = -((positive_sim - max_positive).exp().sum().log() + max_positive - torch.log(mask.sum() + EPSILON))
        else:
            positive_loss = torch.tensor(0.0, device=features.device)

        negative_sim = sim[mask == 0]
        if negative_sim.numel() > 0:
            max_negative = negative_sim.max()
            negative_loss = ((negative_sim - max_negative).exp().sum().log() + max_negative - torch.log((mask.shape[0] * (mask.shape[0] - 1) - mask.sum()) + EPSILON))
        else:
            negative_loss = torch.tensor(0.0, device=features.device)

        loss = positive_loss + negative_loss

        # logging.info(f"orth_loss: {loss}")
        return loss

    def div_loss(self, features: torch.Tensor, domain_ids: torch.Tensor):

        loss = 0.0
        if self.domain_distributions:
            sample_means = []
            batch_size = features.shape[0]
            sample_per_domain = batch_size // len(self.domain_distributions) + 1

            for domain_id, domain_dist in self.domain_distributions.items():
                if isinstance(domain_dist, CovarianceDist):
                    # sample_means.append(domain_dist.generate(sample_per_domain))
                    sample_means.append(domain_dist.mean.unsqueeze(0))
            sample_means = torch.cat(sample_means, dim=0).to(self._device, non_blocking=True)
            if sample_means.shape[0] > batch_size:
                shuffle_idx = torch.randperm(sample_means.shape[0])[:batch_size]
                sample_means = sample_means[shuffle_idx]

            features = F.normalize(features, p=2, dim=1, eps=EPSILON) # batch_size, embedding_dim
            sample_means = F.normalize(sample_means, p=2, dim=1, eps=EPSILON) # batch_size, embedding_dim

            if torch.isnan(features).any() or torch.isinf(features).any():
                raise ValueError("Features contain NaN or inf values.")
            if torch.isnan(sample_means).any() or torch.isinf(sample_means).any():
                raise ValueError("Sample means contain NaN or inf values.")

            consine_sim = torch.mm(features, sample_means.t())
            loss = - torch.mean(consine_sim)

        # logging.info(f"div_loss: {loss}")
        return loss