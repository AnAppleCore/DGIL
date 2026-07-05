import copy
import logging

import numpy as np
import torch
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import MSLNet
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args: dict):
        super().__init__(args)
        self._network = MSLNet(args, pretrained=True)
        self.batch_size = args["batch_size"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.weight_decay = args["weight_decay"]
        self.milestones = args["milestones"]
        self.bcb_lrscale = args.get("bcb_lrscale", 1.0 / 100)
        self.fix_bcb = self.bcb_lrscale == 0
        self.kd_lambda = args.get("kd_lambda", 1e-3)
        self.kd_T = args.get("kd_T", 2)
        self.ema_gamma = args.get("ema_gamma", 0.96)
        self.ce_current_task_only = args.get("ce_current_task_only", True)
        self.use_memory = args.get("memory_size", 0) > 0 or args.get("memory_per_class", 0) > 0
        self.head_lrscale = args.get("head_lrscale", 1.0)
        self.grad_clip_norm = args.get("grad_clip_norm", None)
        self.train_logit_norm = args.get("train_logit_norm", None)
        self.eval_logit_norm = args.get("eval_logit_norm", self.train_logit_norm)
        self.ca_epochs = args.get("ca_epochs", 0)
        self.ca_samples_per_class = args.get("ca_samples_per_class", 256)
        self.ca_lr = args.get("ca_lr", self.lrate * self.head_lrscale)
        self.ca_logit_norm = args.get("ca_logit_norm", self.train_logit_norm)

        self.seed = args["seed"]
        self.task_sizes = []
        self._ema_network = None
        self._cur_domain = 0
        self.cls_to_task_id = {}
        self.cls_to_domain_id = {}
        self.domain_id_to_cls = {}

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
        self._refresh_ema_network()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + task_size
        self.topk = min(self.topk, self._total_classes)
        self._network.update_fc(self._total_classes)

        try:
            self._cur_domain = data_manager.get_cur_domain(self._cur_task)
        except Exception:
            self._cur_domain = 0
        for c_id in range(self._known_classes, self._total_classes):
            self.cls_to_task_id[c_id] = self._cur_task
            self.cls_to_domain_id[c_id] = self._cur_domain
            if self._cur_domain not in self.domain_id_to_cls:
                self.domain_id_to_cls[self._cur_domain] = []
            self.domain_id_to_cls[self._cur_domain].append(c_id)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self._network.to(self._device)
        if self._ema_network is not None:
            self._ema_network.to(self._device)
            self._ema_network.eval()

        appendent = self._get_memory() if self.use_memory else []
        train_dset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=appendent,
        )
        test_dset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.train_loader = DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
        )

        self._stage1_training(self.train_loader, self.test_loader)
        self._compute_distributions(data_manager)
        if self._cur_task > 0 and self.ca_epochs > 0:
            self._stage2_calibrate_classifier()

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
            if self._ema_network is not None and isinstance(self._ema_network, nn.DataParallel):
                self._ema_network = self._ema_network.module

        if self.use_memory:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _stage1_training(self, train_loader, test_loader):
        base_params = self._network.backbone.parameters()
        head_params = [p for p in self._network.fc.parameters() if p.requires_grad]
        if not self.fix_bcb:
            network_params = [
                {
                    "params": base_params,
                    "lr": self.lrate * self.bcb_lrscale,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": head_params,
                    "lr": self.lrate * self.head_lrscale,
                    "weight_decay": self.weight_decay,
                },
            ]
        else:
            for p in base_params:
                p.requires_grad = False
            network_params = [
                {"params": head_params, "lr": self.lrate * self.head_lrscale, "weight_decay": self.weight_decay}
            ]

        optimizer = optim.SGD(
            network_params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            if self._ema_network is not None:
                self._ema_network = nn.DataParallel(self._ema_network, self._multiple_gpus)

        self._run(train_loader, test_loader, optimizer, scheduler)

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            ce_losses = 0.0
            kd_losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"][:, : self._total_classes]
                logits = self._normalize_logits_by_task(logits, self.train_logit_norm)
                if (self.use_memory and self._cur_task > 0) or not self.ce_current_task_only:
                    loss_ce = F.cross_entropy(logits, targets)
                    acc_logits = logits
                    acc_targets = targets
                else:
                    acc_targets = targets - self._known_classes
                    acc_logits = logits[:, self._known_classes : self._total_classes]
                    loss_ce = F.cross_entropy(acc_logits, acc_targets)
                loss_kd = self._compute_kd_loss(inputs, logits)
                loss = loss_ce + self.kd_lambda * loss_kd

                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.grad_clip_norm)
                optimizer.step()
                self._update_ema_network()

                losses += loss.item()
                ce_losses += loss_ce.item()
                kd_losses += loss_kd.item()
                _, preds = torch.max(acc_logits, dim=1)
                correct += preds.eq(acc_targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = (
                    "Task {}, Epoch {}/{} => Loss {:.3f}, CE {:.3f}, KD {:.3f}, "
                    "Train_accy {:.2f}, Test_accy {:.2f}"
                ).format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    ce_losses / len(train_loader),
                    kd_losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = (
                    "Task {}, Epoch {}/{} => Loss {:.3f}, CE {:.3f}, KD {:.3f}, "
                    "Train_accy {:.2f}"
                ).format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    ce_losses / len(train_loader),
                    kd_losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _normalize_logits_by_task(self, logits, norm_mode):
        if norm_mode is None or norm_mode == "none" or self._cur_task == 0:
            return logits
        if norm_mode != "l2":
            raise ValueError(f"Unknown logit normalization mode: {norm_mode}")

        task_logits = []
        start = 0
        for task_size in self.task_sizes:
            end = start + task_size
            task_logit = logits[:, start:end]
            task_norm = torch.norm(task_logit, p=2, dim=1, keepdim=True).clamp_min(1e-7)
            task_logits.append(task_logit / task_norm)
            start = end
        return torch.cat(task_logits, dim=1)

    def _compute_kd_loss(self, inputs, logits):
        if self._cur_task == 0 or self._ema_network is None or self._known_classes == 0:
            return logits.new_tensor(0.0)
        with torch.no_grad():
            teacher_logits = self._ema_network(inputs)["logits"][:, : self._known_classes]
            teacher_logits = self._normalize_logits_by_task(teacher_logits, self.train_logit_norm)
            soft_targets = torch.softmax(teacher_logits / self.kd_T, dim=1)
        student_log_probs = torch.log_softmax(
            logits[:, : self._known_classes] / self.kd_T, dim=1
        )
        return -torch.sum(soft_targets * student_log_probs, dim=1).mean()

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"][:, : self._total_classes]
                outputs = self._normalize_logits_by_task(outputs, self.eval_logit_norm)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"][:, : self._total_classes]
                outputs = self._normalize_logits_by_task(outputs, self.eval_logit_norm)
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _compute_distributions(self, data_manager):
        if hasattr(self, "_class_means_msl") and self._class_means_msl is not None:
            old_classes = self._class_means_msl.shape[0]
            assert old_classes == self._known_classes
            class_means = np.zeros((self._total_classes, self.feature_dim))
            class_means[: self._known_classes] = self._class_means_msl
            self._class_means_msl = class_means

            class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            class_covs[: self._known_classes] = self._class_covs_msl
            self._class_covs_msl = class_covs
        else:
            self._class_means_msl = np.zeros((self._total_classes, self.feature_dim))
            self._class_covs_msl = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

        for class_idx in range(self._known_classes, self._total_classes):
            _, _, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            class_mean = np.mean(vectors, axis=0)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)
            class_cov = class_cov + torch.eye(class_mean.shape[-1], dtype=torch.float64) * 1e-4
            self._class_means_msl[class_idx, :] = class_mean
            self._class_covs_msl[class_idx, ...] = class_cov.float()

        logging.info("Compute MSL distributions for classes {}-{}".format(self._known_classes, self._total_classes))

    def _stage2_calibrate_classifier(self):
        for p in self._network.backbone.parameters():
            p.requires_grad = False
        for p in self._network.fc.parameters():
            p.requires_grad = True

        self._network.to(self._device)
        was_parallel = False
        if len(self._multiple_gpus) > 1 and not isinstance(self._network, nn.DataParallel):
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            was_parallel = True

        fc_params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = optim.SGD(fc_params, lr=self.ca_lr, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.ca_epochs)
        crct_num = self._total_classes

        for epoch in range(self.ca_epochs):
            sampled_data = []
            sampled_label = []
            for class_idx in range(crct_num):
                cls_mean = torch.tensor(self._class_means_msl[class_idx], dtype=torch.float32).to(self._device)
                cls_cov = self._class_covs_msl[class_idx].to(self._device)
                distribution = MultivariateNormal(cls_mean, cls_cov)
                sampled_data.append(distribution.sample(sample_shape=(self.ca_samples_per_class,)))
                sampled_label.extend([class_idx] * self.ca_samples_per_class)

            inputs = torch.cat(sampled_data, dim=0).float().to(self._device)
            targets = torch.tensor(sampled_label).long().to(self._device)
            shuffle_index = torch.randperm(inputs.size(0), device=self._device)
            inputs = inputs[shuffle_index]
            targets = targets[shuffle_index]

            self._network.eval()
            losses = 0.0
            batch_size = self.batch_size
            num_batches = int(np.ceil(inputs.size(0) / batch_size))
            for batch_idx in range(num_batches):
                batch_inputs = inputs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_targets = targets[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
                logits = net.fc(batch_inputs)["logits"][:, :crct_num]
                logits = self._normalize_logits_by_task(logits, self.ca_logit_norm)
                loss = F.cross_entropy(logits, batch_targets)

                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.fc.parameters(), self.grad_clip_norm)
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = "CA Task {} Epoch {}/{} => Loss {:.3f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.ca_epochs,
                losses / max(num_batches, 1),
                test_acc,
            )
            logging.info(info)

        if was_parallel:
            self._network = self._network.module
        for p in self._network.backbone.parameters():
            p.requires_grad = True

    def _refresh_ema_network(self):
        self._ema_network = copy.deepcopy(self._network)
        self._ema_network.to(self._device)
        self._ema_network.eval()
        for p in self._ema_network.parameters():
            p.requires_grad = False

    def _update_ema_network(self):
        if self._ema_network is None:
            return
        current = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        teacher = self._ema_network.module if isinstance(self._ema_network, nn.DataParallel) else self._ema_network
        with torch.no_grad():
            current_state = current.state_dict()
            teacher_state = teacher.state_dict()
            for name, teacher_value in teacher_state.items():
                current_value = current_state.get(name)
                if current_value is None:
                    continue
                if current_value.shape == teacher_value.shape:
                    if not torch.is_floating_point(teacher_value):
                        teacher_value.copy_(current_value)
                    else:
                        teacher_value.mul_(self.ema_gamma).add_(current_value, alpha=1.0 - self.ema_gamma)
                    continue
                if name in {"fc.metric", "fc.bias"} and current_value.shape[0] >= teacher_value.shape[0]:
                    current_slice = current_value[: teacher_value.shape[0]]
                    teacher_value.mul_(self.ema_gamma).add_(current_slice, alpha=1.0 - self.ema_gamma)
