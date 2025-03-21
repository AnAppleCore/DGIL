import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,MultiBranchCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

# tune the model at first session with adapter, and then conduct simplecil.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self. batch_size = args["batch_size"]
        self. init_lr = args["init_lr"]
        
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self.dot_epochs = args.get('dot_epochs', 0)
        self._cur_domain = 0
        self.cls_to_task_id = {}
        self.cls_to_domain_id = {}
        self.domain_id_to_cls = {}

    def after_task(self):
        self._known_classes = self._total_classes

    def replace_fc(self, trainloader, model, args):       
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data)
                if len(self.domain_id_to_cls.keys()) > 1 and self.dot_epochs > 0:
                    fake_inps, fake_tgts = [], []
                    for d_id, cid_list in self.domain_id_to_cls.items():
                        ptp_inp = []
                        num_sampled_pcid = 256//len(cid_list)
                        for c_id in cid_list:
                            cls_mean = torch.tensor(self._class_means_slca[c_id], dtype=torch.float64).to(self._device)
                            cls_cov = self._class_covs_slca[c_id].to(self._device)
                            m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                            ptp_inp.append(m.sample(sample_shape=(num_sampled_pcid,)))
                        ptp_inp = torch.cat(ptp_inp, dim=0)

                        with torch.no_grad():
                            fake_inp = self._network.domain_tsf(embedding, ptp_inp, ptp_inp)
                            fake_tgt = torch.zeros_like(label) + label
                        fake_inps.append(fake_inp)
                        fake_tgts.append(fake_tgt.detach())

                    fake_inps = torch.cat(fake_inps, dim=0)
                    fake_tgts = torch.cat(fake_tgts, dim=0)
                    embedding = torch.cat([embedding, fake_inps], dim=0)
                    label = torch.cat([label, fake_tgts], dim=0)

                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Y = target2onehot(label_list, self.args["nb_classes"])
        Features_h = F.relu(embedding_list @ self.W_rand.cpu())
        self.Q = self.Q + Features_h.T @ Y
        self.G = self.G + Features_h.T @ Features_h
        ridge = self.optimise_ridge_parameter(Features_h, Y)
        Wo = torch.linalg.solve(self.G + ridge*torch.eye(self.G.size(dim=0)), self.Q).T # better nmerical stability than .invv
        self._network.fc.weight.data = Wo[0:self._network.fc.weight.shape[0],:].to(self._device)
        
        return model

    def setup_RP(self):
        M = self.args['M']
        self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(self._device)).requires_grad_(False) # num classes in task x M
        self._network.RP_dim = M
        self.W_rand = torch.randn(self._network.fc.in_features, M).to(self._device)
        self._network.W_rand = self.W_rand

        self.Q = torch.zeros(M, self.args["nb_classes"])
        self.G = torch.zeros(M, M)

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge*torch.eye(G_val.size(dim=0)), Q_val).T #better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::,:] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        print('selected lambda =',ridge)
        return ridge
    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

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

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)
        
        if self._cur_task == 0:
            # show total parameters and trainable parameters
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            if total_params != total_trainable_params:
                for name, param in self._network.named_parameters():
                    if param.requires_grad:
                        print(name, param.numel())
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            pass
        if self._cur_task == 0 and self.args["use_RP"]:
            self.setup_RP()

        self._compute_distributions(self.data_manager)
        if len(self.domain_id_to_cls.keys()) > 1 and self.dot_epochs > 0:
            self._stage2_domain_transformation()

        self.replace_fc(train_loader_for_protonet, self._network, None)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

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

        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            # vectors = np.concatenate([vectors_aug, vectors])

            class_mean = np.mean(vectors, axis=0)
            # class_cov = np.cov(vectors.T)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(class_mean.shape[-1])*1e-4
            self._class_means_slca[class_idx, :] = class_mean
            self._class_covs_slca[class_idx, ...] = class_cov

        logging.info('Compute distributions for classes {}-{}'.format(self._known_classes, self._total_classes))


    def _stage2_domain_transformation(self):
        run_epochs = self.dot_epochs
        crct_num = self._total_classes
        if self._network.domain_clf is None:
            self._network.reset_domain_tsf_clf(
                nb_classes=512, num_domains=512
            )
        param_list = {}
        for n, p in self._network.named_parameters():
            if "domain" in n or "class" in n:
                p.requires_grad = True
                param_list[n] = p
        network_params = [{'params': param_list.values(), 'lr': 0.01, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=0.01, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=run_epochs)

        self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.eval()
        for epoch in range(run_epochs):
            self._network.train()
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

                    fake_inp = self._network.domain_tsf(inp, ptp_inp, ptp_inp)
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

            scheduler.step()
            info = 'DOT Task {} => Loss {:.3f}, Cls_loss {:.3f}, Dom_loss {:.3f}'.format(
                self._cur_task, losses/self._total_classes, losses_cls/self._total_classes, losses_dom/self._total_classes)
            logging.info(info)


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