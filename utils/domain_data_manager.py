import logging

import numpy as np
from utils.data_manager import DataManager, _get_idata, _map_new_class_index


class DomainDataManager(DataManager):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed, init_cls, increment)

    def _setup_data(self, dataset_name, shuffle, seed, init_cls, increment):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

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

        # By default, we use all the training data and targets from the first domain
        # and for the later domains, we only use the training data and targets of the first task
        self.num_domains = len(self._train_data)
        logging.info("Number of domains: {}".format(self.num_domains))

        _train_data = [self._train_data[0]]
        _train_targets = [self._train_targets[0]]
        _train_domain_idx = [np.zeros(len(self._train_data[0]), dtype=np.int32)]
        _test_domain_idx = [np.zeros(len(self._test_data[0]), dtype=np.int32)]
        logging.info("Number of trainings imgs from domain 0: {}/{}".format(len(self._train_data[0]), len(self._train_data[0])))
        logging.info("Number of test imgs from domain 0: {}/{}".format(len(self._test_data[0]), len(self._test_data[0])))

        for d in range(1, self.num_domains):
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

        #TODO implement other data steam setting
