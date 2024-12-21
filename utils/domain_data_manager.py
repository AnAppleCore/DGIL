import logging

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.data_manager import (DataManager, DummyDataset, _get_idata,
                                _map_new_class_index)


class DomainDataManager(DataManager):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args:dict):
        self.arg = args
        self.dataset_name = dataset_name
        self.enable_dgil = args.get("enable_dgil", False)
        self.reference_domain_id = args.get("reference_domain_id", 0)
        self._setup_data(dataset_name, shuffle, seed, init_cls, increment)

    def _setup_data(self, dataset_name, shuffle, seed, init_cls, increment):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path
        self.num_domains = len(self._train_data)
        self.domain_names = idata.domain_names
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
        logging.info(self._class_order)

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

        # Disable DGIL
        if not self.enable_dgil:
            self._train_domain_idx = []
            self._test_domain_idx = []
            for d in range(self.num_domains):
                self._train_domain_idx.append(np.ones(len(self._train_data[d]), dtype=np.int32) * d)
                self._test_domain_idx.append(np.ones(len(self._test_data[d]), dtype=np.int32) * d)
                logging.info("Number of trainings imgs from domain {}: {}/{}".format(self.domain_names[d], len(self._train_data[d]), len(self._train_data[d])))
                logging.info("Number of test imgs from domain {}: {}/{}".format(self.domain_names[d], len(self._test_data[d]), len(self._test_data[d])))

            self._train_domain_idx = np.concatenate(self._train_domain_idx)
            self._test_domain_idx = np.concatenate(self._test_domain_idx)

            self._train_data = np.concatenate(self._train_data)
            self._train_targets = np.concatenate(self._train_targets)
            self._test_data = np.concatenate(self._test_data)
            self._test_targets = np.concatenate(self._test_targets)

            return

        else:
            _train_data, _train_targets = [], []
            _train_domain_idx, _test_domain_idx = [], []

            for d in range(self.num_domains):
                if d == self.reference_domain_id:
                    _train_data_d = self._train_data[d]
                    _train_targets_d = self._train_targets[d]
                else:
                    # retrieve the first task
                    _train_data_d, _train_targets_d = self._select(
                        self._train_data[d], self._train_targets[d], 0, self.get_task_size(0)
                    )
                _train_data.append(_train_data_d)
                _train_targets.append(_train_targets_d)
                _train_domain_idx.append(np.ones(len(_train_data_d), dtype=np.int32) * d)
                _test_domain_idx.append(np.ones(len(self._test_data[d]), dtype=np.int32) * d)
                logging.info("Number of trainings imgs from domain {}: {}/{}".format(d, len(_train_data_d), len(self._train_data[d])))
                logging.info("Number of test imgs from domain {}: {}/{}".format(d, len(self._test_data[d]), len(self._test_data[d])))

            self._train_data = np.concatenate(_train_data)
            self._train_targets = np.concatenate(_train_targets)
            self._train_domain_idx = np.concatenate(_train_domain_idx)

            self._test_data = np.concatenate(self._test_data)
            self._test_targets = np.concatenate(self._test_targets)
            self._test_domain_idx = np.concatenate(_test_domain_idx)

            return


    def get_domain_dataset(
        self, indices, source, mode, domain_id, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            domain_idx = np.where(self._train_domain_idx == domain_id)[0]
            x, y = self._train_data[domain_idx], self._train_targets[domain_idx]
        elif source == "test":
            domain_idx = np.where(self._test_domain_idx == domain_id)[0]
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