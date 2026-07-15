import logging

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.data_manager import (DataManager, DummyDataset, _get_idata,
                                _map_new_class_index)


class DomainDataManager(DataManager):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args:dict):
        self.args = args
        self.dataset_name = dataset_name
        self.enable_dgil = args.get("enable_dgil", False)
        self.random_reference = args.get("random_reference", False)
        self.reference_domain_id = args.get("reference_domain_id", 0)
        self.multi_domain_base_task = args.get("multi_domain_base_task", False)
        self.domain_incremental = args.get("domain_incremental", False)
        self.domain_groups = args.get("domain_groups", None)
        self.unseen_domain_ids = args.get("unseen_domain_ids", None)
        self._setup_data(dataset_name, shuffle, seed, init_cls, increment)

    def _setup_data(self, dataset_name, shuffle, seed, init_cls, increment):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path
        self.num_domains = len(self._train_data)
        self.domain_names = idata.domain_names
        assert self.num_domains == len(self.domain_names), "Number of domains and domain names do not match."
        logging.info("Number of domains: {}".format(self.num_domains))

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets[0])))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info("Class order: {}".format(self._class_order))

        # Increments
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

        # Map indices
        self._train_targets = [
            _map_new_class_index(_train_targets_d, self._class_order)
            for _train_targets_d in self._train_targets
        ]
        self._test_targets = [
            _map_new_class_index(_test_targets_d, self._class_order)
            for _test_targets_d in self._test_targets
        ]

        unseen = set(self.unseen_domain_ids or [])
        invalid_unseen = sorted(domain_id for domain_id in unseen if domain_id < 0 or domain_id >= self.num_domains)
        if invalid_unseen:
            raise ValueError("unseen_domain_ids contains invalid domain ids: {}".format(invalid_unseen))

        if self.domain_groups is not None:
            self.domain_groups = [list(group) for group in self.domain_groups]
        elif self.domain_incremental:
            self.domain_groups = [[d] for d in range(self.num_domains) if d not in unseen]

        if self.domain_groups is not None:
            if not self.domain_groups or any(not group for group in self.domain_groups):
                raise ValueError("domain_groups must contain at least one non-empty group.")
            flat_domain_ids = [domain_id for group in self.domain_groups for domain_id in group]
            invalid_domain_ids = sorted({
                domain_id for domain_id in flat_domain_ids
                if not isinstance(domain_id, (int, np.integer)) or domain_id < 0 or domain_id >= self.num_domains
            }, key=str)
            if invalid_domain_ids:
                raise ValueError("domain_groups contains invalid domain ids: {}".format(invalid_domain_ids))
            if len(flat_domain_ids) != len(set(flat_domain_ids)):
                raise ValueError("domain_groups must not contain duplicate domain ids.")
            overlap = sorted(set(flat_domain_ids) & unseen)
            if overlap:
                raise ValueError("Training domain_groups overlap unseen_domain_ids: {}".format(overlap))

        if self.domain_groups is not None and self.args.get("random_domain_order", False):
            np.random.seed(seed)
            order = np.random.permutation(len(self.domain_groups)).tolist()
            self.domain_groups = [self.domain_groups[i] for i in order]
        if self.domain_incremental:
            assert len(self._class_order) == init_cls, "Domain-incremental mode expects fixed classes; set init_cls to all classes."
            self._increments = [init_cls] + [0] * (len(self.domain_groups) - 1)
            logging.info("Domain-incremental domain groups: {}".format(self.domain_groups))

            self._original_train_data = np.concatenate(self._train_data)
            self._original_train_targets = np.concatenate(self._train_targets)
            self._original_train_domain_idx = []
            for d in range(self.num_domains):
                self._original_train_domain_idx.append(np.ones(len(self._train_data[d]), dtype=np.int32) * d)
            self._original_train_domain_idx = np.concatenate(self._original_train_domain_idx)

            _train_data, _train_targets, _train_domain_idx = [], [], []
            for task_id, domain_group in enumerate(self.domain_groups):
                logging.info("Task {}: training domain group {}".format(task_id, domain_group))
                for d in domain_group:
                    _train_data.append(self._train_data[d])
                    _train_targets.append(self._train_targets[d])
                    _train_domain_idx.append(np.ones(len(self._train_data[d]), dtype=np.int32) * d)

            _test_domain_idx = []
            for d in range(self.num_domains):
                _test_domain_idx.append(np.ones(len(self._test_data[d]), dtype=np.int32) * d)
                logging.info("Number of trainings imgs from domain [{}] {}: {}/{}".format(d, self.domain_names[d], len(self._train_data[d]), len(self._train_data[d])))
                logging.info("Number of test imgs from domain [{}] {}: {}/{}".format(d, self.domain_names[d], len(self._test_data[d]), len(self._test_data[d])))

            self.ref_domain_ids = [group[0] for group in self.domain_groups]
            self._train_data = np.concatenate(_train_data)
            self._train_targets = np.concatenate(_train_targets)
            self._train_domain_idx = np.concatenate(_train_domain_idx)
            self._test_data = np.concatenate(self._test_data)
            self._test_targets = np.concatenate(self._test_targets)
            self._test_domain_idx = np.concatenate(_test_domain_idx)
            return

        # Disable DGIL
        if not self.enable_dgil:
            self._train_domain_idx = []
            self._test_domain_idx = []
            for d in range(self.num_domains):
                self._train_domain_idx.append(np.ones(len(self._train_data[d]), dtype=np.int32) * d)
                self._test_domain_idx.append(np.ones(len(self._test_data[d]), dtype=np.int32) * d)
                logging.info("Number of trainings imgs from domain [{}] {}: {}/{}".format(d, self.domain_names[d], len(self._train_data[d]), len(self._train_data[d])))
                logging.info("Number of test imgs from domain [{}] {}: {}/{}".format(d, self.domain_names[d], len(self._test_data[d]), len(self._test_data[d])))

            self._train_domain_idx = np.concatenate(self._train_domain_idx)
            self._test_domain_idx = np.concatenate(self._test_domain_idx)

            self._train_data = np.concatenate(self._train_data)
            self._train_targets = np.concatenate(self._train_targets)
            self._test_data = np.concatenate(self._test_data)
            self._test_targets = np.concatenate(self._test_targets)

            self.original_train_data = self._train_data
            self.original_train_targets = self._train_targets
            self.original_train_domain_idx = self._train_domain_idx

            return

        else:
            _train_data, _train_targets = [], []
            _train_domain_idx, _test_domain_idx = [], []

            # set training data and targets
            if self.domain_groups is not None:
                train_domain_groups = self.domain_groups
                assert len(train_domain_groups) == self.nb_tasks, "domain_groups must match the number of tasks."
                self.ref_domain_ids = [group[0] for group in train_domain_groups]
            elif self.random_reference:
                np.random.seed(seed)
                self.ref_domain_ids = self.assign_domain_id()
                train_domain_groups = [[domain_id] for domain_id in self.ref_domain_ids]
            else:
                self.ref_domain_ids = [self.reference_domain_id] * self.nb_tasks
                train_domain_groups = [[self.reference_domain_id] for _ in range(self.nb_tasks)]

            for task_id in range(self.nb_tasks):
                domain_group = train_domain_groups[task_id]
                logging.info("Task {}: reference domain group is {}".format(task_id, domain_group))
                for ref_domain_id in domain_group:
                    _train_data_t, _train_targets_t = self._select(
                        self._train_data[ref_domain_id], self._train_targets[ref_domain_id],
                        sum(self._increments[:task_id]), sum(self._increments[:task_id+1])
                    )
                    _train_data.append(_train_data_t)
                    _train_targets.append(_train_targets_t)
                    _train_domain_idx.append(np.ones(len(_train_data_t), dtype=np.int32) * ref_domain_id)

            if self.multi_domain_base_task:
                for d in range(self.num_domains):
                    if d != self.ref_domain_ids[0]:
                        _train_data_d, _train_targets_d = self._select(
                            self._train_data[d], self._train_targets[d], 0, self.get_task_size(0)
                        )
                        _train_data.append(_train_data_d)
                        _train_targets.append(_train_targets_d)
                        _train_domain_idx.append(np.ones(len(_train_data_d), dtype=np.int32) * d)

            self._train_domain_idx = np.concatenate(_train_domain_idx)

            for d in range(self.num_domains):
                _test_domain_idx.append(np.ones(len(self._test_data[d]), dtype=np.int32) * d)
                logging.info("Number of trainings imgs from domain [{}] {}: {}/{}".format(d, self.domain_names[d], len(np.where(self._train_domain_idx == d)[0]), len(self._train_data[d])))
                logging.info("Number of test imgs from domain [{}] {}: {}/{}".format(d, self.domain_names[d], len(self._test_data[d]), len(self._test_data[d])))

            self._original_train_data = np.concatenate(self._train_data)
            self._original_train_targets = np.concatenate(self._train_targets)
            self._original_train_domain_idx = []
            for d in range(self.num_domains):
                self._original_train_domain_idx.append(np.ones(len(self._train_data[d]), dtype=np.int32) * d)
            self._original_train_domain_idx = np.concatenate(self._original_train_domain_idx)

            self._train_data = np.concatenate(_train_data)
            self._train_targets = np.concatenate(_train_targets)
            self._test_data = np.concatenate(self._test_data)
            self._test_targets = np.concatenate(self._test_targets)
            self._test_domain_idx = np.concatenate(_test_domain_idx)

            logging.info("Number of trainings imgs: {}".format(len(self._train_data)))
            logging.info("Number of trainings targets: {}".format(len(self._train_targets)))
            logging.info("Number of trainings domain idx: {}".format(len(self._train_domain_idx)))
            logging.info("Number of test imgs: {}".format(len(self._test_data)))
            logging.info("Number of test targets: {}".format(len(self._test_targets)))
            logging.info("Number of test domain idx: {}".format(len(self._test_domain_idx)))

            return


    def get_domain_incremental_dataset(self, task_id, mode, source="train", ret_data=False):
        if not self.domain_incremental:
            raise ValueError("get_domain_incremental_dataset is only available in domain_incremental mode.")
        if source == "train":
            domain_ids = self.domain_groups[task_id]
            return self.get_domain_dataset(
                np.arange(0, len(self._class_order)), source="train", mode=mode, domain_id=domain_ids, ret_data=ret_data
            )
        if source == "test":
            seen_domain_ids = [d for group in self.domain_groups[:task_id + 1] for d in group]
            return self.get_domain_dataset(
                np.arange(0, len(self._class_order)), source="test", mode=mode, domain_id=seen_domain_ids, ret_data=ret_data
            )
        if source == "test_all":
            return self.get_domain_dataset(
                np.arange(0, len(self._class_order)), source="test", mode=mode, domain_id="all", ret_data=ret_data
            )
        raise ValueError("Unknown data source {}.".format(source))

    def get_domain_dataset(
        self, indices, source, mode, domain_id, appendent=None, ret_data=False, m_rate=None, exclude=False
    ):
        if source == "train":
            if type(domain_id) == list:
                domain_idx = np.where(np.isin(self._original_train_domain_idx, domain_id))[0]
            elif type(domain_id) == int:
                if exclude:
                    domain_idx = np.where(self._original_train_domain_idx != domain_id)[0]
                else:
                    domain_idx = np.where(self._original_train_domain_idx == domain_id)[0]
            elif domain_id == 'all':
                domain_idx = np.arange(len(self._original_train_data))
            else:
                raise ValueError("Unknown domain_id type {}.".format(type(domain_id)))
            x, y = self._original_train_data[domain_idx], self._original_train_targets[domain_idx]
        elif source == "test":
            if type(domain_id) == list:
                domain_idx = np.where(np.isin(self._test_domain_idx, domain_id))[0]
            elif type(domain_id) == int:
                if exclude:
                    domain_idx = np.where(self._test_domain_idx != domain_id)[0]
                else:
                    domain_idx = np.where(self._test_domain_idx == domain_id)[0]
            elif domain_id == 'all':
                domain_idx = np.arange(len(self._test_data))
            else:
                raise ValueError("Unknown domain_id type {}.".format(type(domain_id)))
            x, y = self._test_data[domain_idx], self._test_targets[domain_idx]
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)
        

    def assign_domain_id(self):

        num_unseen_domain = self.args.get("num_unseen_domain", 0)
        assert num_unseen_domain < self.num_domains

        if num_unseen_domain == 0:
            # Create an array with repeated domain IDs
            domain_ids = np.tile(np.arange(self.num_domains), self.nb_tasks // self.num_domains)
            
            # If there are extra tasks, randomly assign the remainder
            domain_ids = np.concatenate([
                domain_ids, np.random.choice(np.arange(self.num_domains), self.nb_tasks % self.num_domains, replace=False)
            ])
            
            # Shuffle the domain IDs to randomize the assignments
            np.random.shuffle(domain_ids)

        elif num_unseen_domain > 0:
            domain_list = np.arange(self.num_domains)
            num_seen_domain = self.num_domains - num_unseen_domain
            seen_domain_list = np.random.choice(domain_list, num_seen_domain, replace=False)
            unseen_domain_list = np.setdiff1d(domain_list, seen_domain_list)

            domain_ids = np.tile(seen_domain_list, self.nb_tasks // num_seen_domain)
            domain_ids = np.concatenate([
                domain_ids, np.random.choice(seen_domain_list, self.nb_tasks % num_seen_domain, replace=False)
            ])
            np.random.shuffle(domain_ids)
        
        return domain_ids


    def get_cur_domain(self, task_id):
        return self.ref_domain_ids[task_id]