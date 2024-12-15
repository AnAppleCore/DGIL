import os

import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels, split_train_val


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("/data/datasets/CIFAR", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("/data/datasets/CIFAR", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("/data/datasets/CIFAR", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("/data/datasets/CIFAR", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/data/datasets/ImageNet/train/"
        test_dir = "/data/datasets/ImageNet/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/data/datasets/ImageNet-100/train/"
        test_dir = "/data/datasets/ImageNet-100/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iGanFake(iData):
    use_path = True
    pass


class iCORe50(iData):
    use_path = False
    pass


class iDomainNet(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(345).tolist()
    domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = "/data/datasets/DomainNet/"

        self.train_data = []
        self.train_targets = []

        train_image_list_paths = [
            os.path.join(root_dir, d + "_" + "train" + ".txt") for d in self.domain_names
        ]
        for domain_id, train_image_list_path in enumerate(train_image_list_paths):
            train_image_list = open(train_image_list_path, "r").readlines()
            train_images = [
                os.path.join(root_dir, line.split()[0]) for line in train_image_list
            ]
            train_labels = [
                int(line.split()[1]) for line in train_image_list
            ]

            self.train_data.append(np.array(train_images))
            self.train_targets.append(np.array(train_labels))

        self.test_data = []
        self.test_targets = []

        test_image_list_paths = [
            os.path.join(root_dir, d + "_" + "test" + ".txt") for d in self.domain_names
        ]
        for domain_id, test_image_list_path in enumerate(test_image_list_paths):
            test_image_list = open(test_image_list_path, "r").readlines()
            test_images = [
                os.path.join(root_dir, line.split()[0]) for line in test_image_list
            ]
            test_labels = [
                int(line.split()[1]) for line in test_image_list
            ]

            self.test_data.append(np.array(test_images))
            self.test_targets.append(np.array(test_labels))
            

class iOfficeHome(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(65).tolist()
    domain_names = ["Art", "Clipart", "Product", "Real World"]

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = "/data/datasets/OfficeHomeDataset_10072016/"

        self.train_data = []
        self.train_targets = []

        self.test_data = []
        self.test_targets = []

        for domain_id, domain_name in enumerate(self.domain_names):
            domain_img_dir = os.path.join(root_dir, domain_name)
            domain_dset = datasets.ImageFolder(domain_img_dir)
            domain_images, domain_labels = split_images_labels(domain_dset.imgs)
            train_data_d, train_targets_d, test_data_d, test_targets_d = \
                split_train_val(domain_images, domain_labels, val_ratio=0.3, seed=42)

            self.train_data.append(train_data_d)
            self.train_targets.append(train_targets_d)
            self.test_data.append(test_data_d)
            self.test_targets.append(test_targets_d)


class iOffice31(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(31).tolist()
    domain_names = ["amazon", "dslr", "webcam"]

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = "/data/datasets/office31/"

        self.train_data = []
        self.train_targets = []

        self.test_data = []
        self.test_targets = []

        for domain_id, domain_name in enumerate(self.domain_names):
            domain_img_dir = os.path.join(root_dir, domain_name)
            domain_dset = datasets.ImageFolder(domain_img_dir)
            domain_images, domain_labels = split_images_labels(domain_dset.imgs)
            train_data_d, train_targets_d, test_data_d, test_targets_d = \
                split_train_val(domain_images, domain_labels, val_ratio=0.3, seed=42)

            self.train_data.append(train_data_d)
            self.train_targets.append(train_targets_d)
            self.test_data.append(test_data_d)
            self.test_targets.append(test_targets_d)


class iOfficeCaltech(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(10).tolist()
    domain_names = ["amazon", "caltech", "dslr", "webcam"]

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = "/data/datasets/office_caltech_10/"

        self.train_data = []
        self.train_targets = []

        self.test_data = []
        self.test_targets = []

        for domain_id, domain_name in enumerate(self.domain_names):
            domain_img_dir = os.path.join(root_dir, domain_name)
            domain_dset = datasets.ImageFolder(domain_img_dir)
            domain_images, domain_labels = split_images_labels(domain_dset.imgs)
            train_data_d, train_targets_d, test_data_d, test_targets_d = \
                split_train_val(domain_images, domain_labels, val_ratio=0.3, seed=42)

            self.train_data.append(train_data_d)
            self.train_targets.append(train_targets_d)
            self.test_data.append(test_data_d)
            self.test_targets.append(test_targets_d)


class iImageCLEF(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(12).tolist()
    domain_names = ['i', 'p', 'c'] # b?

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = "/data/datasets/image_CLEF/"

        self.train_data = []
        self.train_targets = []

        self.test_data = []
        self.test_targets = []

        for domain_id, domain_name in enumerate(self.domain_names):
            domain_img_dir = os.path.join(root_dir, domain_name)
            domain_dset = datasets.ImageFolder(domain_img_dir)
            domain_images, domain_labels = split_images_labels(domain_dset.imgs)
            train_data_d, train_targets_d, test_data_d, test_targets_d = \
                split_train_val(domain_images, domain_labels, val_ratio=0.3, seed=42)

            self.train_data.append(train_data_d)
            self.train_targets.append(train_targets_d)
            self.test_data.append(test_data_d)
            self.test_targets.append(test_targets_d)