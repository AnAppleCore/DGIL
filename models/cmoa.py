import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import CMoANet
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if "cmoa" not in args["backbone_type"]:
            raise NotImplementedError("CMoA requires a CMoA backbone")
        self._network = CMoANet(args, pretrained=True)
        self.batch_size = args["batch_size"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.weight_decay = args["weight_decay"]
        self.min_lr = args.get("min_lr", 1e-8)
        self.zeta = args.get("cmoa_zeta", 0.1)
        self.proto_lambda = args.get("cmoa_proto_lambda", 0.1)
        self.proto_temperature = args.get("cmoa_proto_temperature", 0.07)
        self.grad_clip_norm = args.get("grad_clip_norm", None)
        self.kd_weight = args.get("cmoa_kd_weight", 1.0)
        self.kd_temperature = args.get("cmoa_kd_temperature", 2.0)
        self.use_memory = args.get("memory_size", 0) > 0 or args.get("memory_per_class", 0) > 0
        self._class_prototypes = None
        self._old_network = None
        self._cur_domain = 0

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
        proto_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test"
        )

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        proto_loader = DataLoader(proto_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.to(self._device)
        self._stage1_training(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        self._update_prototypes(proto_loader)
        if self.use_memory:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _stage1_training(self, train_loader, test_loader):
        self._set_trainable_parameters()
        trainable_params = [p for p in self._network.parameters() if p.requires_grad]
        optimizer = optim.SGD(trainable_params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.min_lr)
        prog_bar = tqdm(range(self.epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, ce_losses, cos_losses, proto_losses, kd_losses = 0.0, 0.0, 0.0, 0.0, 0.0
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"][:, : self._total_classes]
                features = outputs["features"]
                cur_logits = logits[:, self._known_classes : self._total_classes]
                cur_targets = targets - self._known_classes
                loss_ce = F.cross_entropy(cur_logits, cur_targets)
                loss_cos = outputs.get("moa_loss", logits.new_tensor(0.0))
                loss_proto = self._prototype_contrastive_loss(features, targets)
                loss_kd = self._distillation_loss(logits, inputs)
                loss = loss_ce + self.zeta * loss_cos + self.proto_lambda * loss_proto + self.kd_weight * loss_kd

                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.grad_clip_norm)
                optimizer.step()

                losses += loss.item()
                ce_losses += loss_ce.item()
                cos_losses += loss_cos.item()
                proto_losses += loss_proto.item()
                kd_losses += loss_kd.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = (
                    "Task {}, Epoch {}/{} => Loss {:.3f}, CE {:.3f}, Cos {:.3f}, Proto {:.3f}, KD {:.3f}, "
                    "Train_accy {:.2f}, Test_accy {:.2f}"
                ).format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    ce_losses / len(train_loader),
                    cos_losses / len(train_loader),
                    proto_losses / len(train_loader),
                    kd_losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = (
                    "Task {}, Epoch {}/{} => Loss {:.3f}, CE {:.3f}, Cos {:.3f}, Proto {:.3f}, KD {:.3f}, "
                    "Train_accy {:.2f}"
                ).format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    ce_losses / len(train_loader),
                    cos_losses / len(train_loader),
                    proto_losses / len(train_loader),
                    kd_losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _set_trainable_parameters(self):
        for name, param in self._network.named_parameters():
            is_classifier = name.startswith("fc.") or name.startswith("module.fc.")
            is_moa_adapter = "adaptmlp" in name
            param.requires_grad = is_classifier or is_moa_adapter

    def _prototype_contrastive_loss(self, features, targets):
        proto_bank = features.new_zeros(self._total_classes, features.size(1))
        valid_proto = torch.zeros(self._total_classes, dtype=torch.bool, device=features.device)
        if self._class_prototypes is not None and self._known_classes > 0:
            proto_bank[: self._known_classes] = self._class_prototypes[: self._known_classes].to(features.device)
            valid_proto[: self._known_classes] = True

        for class_idx in torch.unique(targets).tolist():
            class_mask = targets == class_idx
            if torch.any(class_mask):
                proto_bank[class_idx] = features[class_mask].mean(dim=0)
                valid_proto[class_idx] = True

        if valid_proto.sum() <= 1:
            return features.new_tensor(0.0)
        valid_indices = torch.where(valid_proto)[0]
        proto = F.normalize(proto_bank[valid_indices], dim=1)
        query = F.normalize(features, dim=1)
        logits = torch.matmul(query, proto.T) / self.proto_temperature
        target_positions = torch.searchsorted(valid_indices, targets)
        return F.cross_entropy(logits, target_positions)

    def _distillation_loss(self, logits, inputs):
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

    def _update_prototypes(self, loader):
        self._network.eval()
        features, labels = [], []
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                outputs = self._network(inputs)
                features.append(outputs["features"].cpu())
                labels.append(targets.cpu())
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        if self._class_prototypes is None:
            self._class_prototypes = torch.zeros(self._total_classes, features.size(1))
        elif self._class_prototypes.size(0) < self._total_classes:
            new_proto = torch.zeros(self._total_classes, self._class_prototypes.size(1))
            new_proto[: self._class_prototypes.size(0)] = self._class_prototypes
            self._class_prototypes = new_proto

        for class_idx in range(self._known_classes, self._total_classes):
            mask = labels == class_idx
            if torch.any(mask):
                self._class_prototypes[class_idx] = features[mask].mean(dim=0)
        logging.info("Updated CMoA prototypes for classes {}-{}".format(self._known_classes, self._total_classes))

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
