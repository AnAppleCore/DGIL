import logging
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import PromptVitNet
from models.base import BaseLearner
from utils.losses import sup_con
from utils.toolkit import tensor2numpy

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        self._network = PromptVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get("min_lr", 1e-8)
        self.args = args

        self.dot_epochs = args.get("dot_epochs", 0)
        self.dot_lr = args.get("dot_lr", 0.001)
        self.domain_centorids = args.get('domain_centorids', 32)

        self.ca_epochs = args.get("ca_epochs", 3 if self.dot_epochs > 0 else 0)
        self.ca_lr = args.get("ca_lr", 0.001)
        self.logit_norm = args.get("logit_norm", 0.1)

        self.task_sizes = []
        self._cur_domain = 0
        self.cls_to_task_id = {}
        self.cls_to_domain_id = {}
        self.domain_id_to_cls = {}
        self.prototypes = None
        self.prototypes_domain_id = None

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

    def after_task(self):
        self._known_classes = self._total_classes
        if self.ca_epochs > 0:
            self._network.restore_head()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        if self.ca_epochs > 0:
            self._network.update_head(task_size)
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
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if self.ca_epochs > 0:
            if isinstance(self._network, nn.DataParallel):
                self._network.module.back_up_head()
            else:
                self._network.back_up_head()
            self._compute_distribusions(data_manager)
            if len(self.domain_id_to_cls.keys()) > 1 and self.dot_epochs > 0:
                self._domain_transformation()
            if self._cur_task > 0:
                self._compact_classifier()
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

    def get_optimizer(self):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr, 
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

    def _compute_distribusions(self, data_manager):
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
        
        all_features = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            if self.dot_epochs > 0:
                vectors, features, _ = self._extract_layerwise_vectors(idx_loader, task_id=self._cur_task, train=True)
                all_features.append(features)
            else:
                vectors, _ = self._extract_vectors(idx_loader, task_id=self._cur_task, train=True)

            class_mean = np.mean(vectors, axis=0)
            # class_cov = np.cov(vectors.T)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(class_mean.shape[-1])*1e-4
            self._class_means_slca[class_idx, :] = class_mean
            self._class_covs_slca[class_idx, ...] = class_cov

        logging.info('Compute distributions for classes {}-{}'.format(self._known_classes, self._total_classes))

        if self.dot_epochs > 0:
            all_features = np.concatenate(all_features, axis=0) # [num_samples, num_layers, feature_dim]
            all_features_mean = np.mean(all_features, axis=1) # [num_samples, feature_dim]
            kmeans = KMeans(n_clusters=self.domain_centorids).fit(all_features_mean)
            feature_centers = kmeans.cluster_centers_
            # find closest prototype for each center
            prototype_idx = []
            all_idx = np.arange(all_features_mean.shape[0])
            for i in range(self.domain_centorids):
                i_mask = (kmeans.labels_ == i)
                i_idx = all_idx[i_mask]
                dist = np.linalg.norm(all_features_mean[i_mask] - feature_centers[i], axis=1)
                prototype_idx.append(i_idx[np.argmin(dist)])

            if self.prototypes is None:
                self.prototypes = all_features[prototype_idx]
                self.prototypes_domain_id = np.zeros(self.domain_centorids, dtype=np.int32) + self._cur_domain
            else:
                self.prototypes = np.concatenate([self.prototypes, all_features[prototype_idx]], axis=0)
                self.prototypes_domain_id = np.concatenate([
                    self.prototypes_domain_id, np.zeros(self.domain_centorids, dtype=np.int32) + self._cur_domain
                ], axis=0)

        logging.info('Compute domain prototypes for domain {}'.format(self._cur_domain))
    
    def _extract_layerwise_vectors(self, loader, task_id=-1, train=False):
        self._network.eval()
        vectors, features, targets = [], [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors, _features = self._network.module.extract_layerwise_vector(
                        _inputs.to(self._device), task_id=task_id, train=train
                    )
                else:
                    _vectors, _features = self._network.extract_layerwise_vector(
                        _inputs.to(self._device), task_id=task_id, train=train
                    )
                
                vectors.append(_vectors)
                features.append(_features)
                targets.append(_targets)

        vectors = np.concatenate(vectors)
        features = np.concatenate(features)
        targets = np.concatenate(targets)

        return vectors, features, targets

    def _extract_vectors(self, loader, task_id=-1, train=False):
        self._network.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(
                            _inputs.to(self._device), task_id=task_id, train=train
                        )
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.extract_vector(
                            _inputs.to(self._device), task_id=task_id, train=train
                        )
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _domain_transformation(self):
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
        # logging.info(f"DoT trainnable params: {param_list.keys()}")
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

                loss = cls_loss + dom_loss

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

    def _compact_classifier(self):
        for p in self._network.rec_head.parameters():
            p.requires_grad=True
        run_epochs = self.ca_epochs
        crct_num = self._total_classes    
        param_list = [p for p in self._network.rec_head.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': self.ca_lr,
                           'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=self.weight_decay)
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

                outputs = self._network(inp, head_only=True)
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