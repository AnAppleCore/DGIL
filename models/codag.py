import copy
import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self._old_network = None
        self._da_network = None
        self.batch_size = args["batch_size"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.weight_decay = args["weight_decay"]
        self.min_lr = args.get("min_lr", 1e-8)
        self.da_steps = args.get("codag_da_steps", 1)
        self.da_lr = args.get("codag_da_lr", self.lrate)
        self.da_micro_batch_size = args.get("codag_da_micro_batch_size", self.batch_size)
        self.pseudo_threshold = args.get("codag_pseudo_threshold", 0.7)
        self.pseudo_weight = args.get("codag_pseudo_weight", 0.5)
        self.kd_weight = args.get("codag_kd_weight", 1.0)
        self.kd_temperature = args.get("codag_kd_temperature", 2.0)
        self.grad_clip_norm = args.get("grad_clip_norm", None)
        self.use_memory = args.get("memory_size", 0) > 0 or args.get("memory_per_class", 0) > 0
        self._cur_domain = 0
        self._last_da_acc = 0.0
        self._last_dg_acc = 0.0

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size
        self.topk = min(self.topk, self._total_classes)
        self._network.update_fc(self._total_classes)

        try:
            self._cur_domain = data_manager.get_cur_domain(self._cur_task)
        except Exception:
            self._cur_domain = 0
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        appendent = self._get_memory() if self.use_memory else []
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=appendent,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)
        self._train_codag(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self.use_memory:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _train_codag(self, train_loader, test_loader):
        self._network.train()
        optimizer = optim.SGD(self._network.parameters(), lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.min_lr)
        prog_bar = tqdm(range(self.epochs))

        for _, epoch in enumerate(prog_bar):
            self._adapt_da_branch(train_loader)
            self._network.train()
            losses, ce_losses, pseudo_losses, kd_losses = 0.0, 0.0, 0.0, 0.0
            correct, total = 0, 0
            pseudo_kept, pseudo_seen = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"][:, : self._total_classes]
                cur_logits = logits[:, self._known_classes : self._total_classes]
                cur_targets = targets - self._known_classes
                loss_ce = F.cross_entropy(cur_logits, cur_targets)
                loss_pseudo, kept, seen = self._pseudo_label_loss(inputs, logits)
                loss_kd = self._distillation_loss(logits, inputs)
                loss = loss_ce + self.pseudo_weight * loss_pseudo + self.kd_weight * loss_kd

                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.grad_clip_norm)
                optimizer.step()

                losses += loss.item()
                ce_losses += loss_ce.item()
                pseudo_losses += loss_pseudo.item()
                kd_losses += loss_kd.item()
                pseudo_kept += kept
                pseudo_seen += seen
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            pseudo_rate = 0.0 if pseudo_seen == 0 else 100.0 * pseudo_kept / pseudo_seen
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                self._last_dg_acc = self._compute_accuracy(self._network, test_loader)
                self._last_da_acc = self._compute_accuracy(self._da_network, test_loader) if self._da_network is not None else 0.0
                info = (
                    "Task {}, Epoch {}/{} => Loss {:.3f}, CE {:.3f}, Pseudo {:.3f}, KD {:.3f}, "
                    "Pseudo_keep {:.1f}%, Train_accy {:.2f}, DG_accy {:.2f}, DA_accy {:.2f}"
                ).format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    ce_losses / len(train_loader),
                    pseudo_losses / len(train_loader),
                    kd_losses / len(train_loader),
                    pseudo_rate,
                    train_acc,
                    self._last_dg_acc,
                    self._last_da_acc,
                )
            else:
                info = (
                    "Task {}, Epoch {}/{} => Loss {:.3f}, CE {:.3f}, Pseudo {:.3f}, KD {:.3f}, "
                    "Pseudo_keep {:.1f}%, Train_accy {:.2f}"
                ).format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    ce_losses / len(train_loader),
                    pseudo_losses / len(train_loader),
                    kd_losses / len(train_loader),
                    pseudo_rate,
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _adapt_da_branch(self, train_loader):
        self._da_network = copy.deepcopy(self._network)
        if isinstance(self._da_network, nn.DataParallel):
            self._da_network = self._da_network.module
        self._da_network.to(self._device)
        self._da_network.train()
        optimizer = optim.SGD(self._da_network.parameters(), lr=self.da_lr, momentum=0.9, weight_decay=self.weight_decay)
        for _ in range(self.da_steps):
            for _, inputs, targets in train_loader:
                for start in range(0, len(targets), self.da_micro_batch_size):
                    mb_inputs = inputs[start : start + self.da_micro_batch_size].to(self._device)
                    mb_targets = targets[start : start + self.da_micro_batch_size].to(self._device)
                    logits = self._da_network(mb_inputs)["logits"][:, : self._total_classes]
                    cur_logits = logits[:, self._known_classes : self._total_classes]
                    cur_targets = mb_targets - self._known_classes
                    loss = F.cross_entropy(cur_logits, cur_targets)
                    kd = self._distillation_loss(logits, mb_inputs, student_is_da=True)
                    optimizer.zero_grad()
                    (loss + self.kd_weight * kd).backward()
                    if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self._da_network.parameters(), self.grad_clip_norm)
                    optimizer.step()
        self._da_network.eval()

    def _pseudo_label_loss(self, inputs, dg_logits):
        if self._da_network is None:
            return dg_logits.new_tensor(0.0), 0, 0
        with torch.no_grad():
            da_logits = self._da_network(inputs)["logits"][:, self._known_classes : self._total_classes]
            probs = F.softmax(da_logits, dim=1)
            conf, pseudo = probs.max(dim=1)
            mask = conf >= self.pseudo_threshold
        if not torch.any(mask):
            return dg_logits.new_tensor(0.0), 0, int(mask.numel())
        dg_cur_logits = dg_logits[:, self._known_classes : self._total_classes]
        return F.cross_entropy(dg_cur_logits[mask], pseudo[mask]), int(mask.sum().item()), int(mask.numel())

    def _distillation_loss(self, logits, inputs, student_is_da=False):
        if self._old_network is None or self._known_classes == 0:
            return logits.new_tensor(0.0)
        old_network = self._old_network.to(self._device)
        old_network.eval()
        with torch.no_grad():
            old_logits = old_network(inputs)["logits"][:, : self._known_classes]
        student_logits = logits[:, : self._known_classes]
        temperature = self.kd_temperature
        return F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(old_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"][:, : self._total_classes]
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
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)
