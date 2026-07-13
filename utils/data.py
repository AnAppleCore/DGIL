import csv
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from utils.toolkit import split_images_labels, split_train_val

DATA_ROOT = os.environ.get("DGIL_DATA_ROOT", "/data/datasets")


def dataset_path(*parts):
    return os.path.join(DATA_ROOT, *parts)


MultiDomainDatasets = [
    "domainnet", "minidomainnet", "officehome", "office31", "officecaltech", "imageclef", "digitsdg", "digitsfive", "core50",
    "rxrx1_balanced300", "rxrx1_balanced100", "rxrx1_balanced50", "fldr", "camelyon17", "midog25"
]


def use_multi_domain_dataset(dataset_name):
    dataset_name = dataset_name.lower()
    return dataset_name in MultiDomainDatasets


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class SelfStandardize(object):
    def __call__(self, tensor):
        if not torch.is_tensor(tensor):
            return tensor
        mean = tensor.mean(dim=(1, 2), keepdim=True)
        std = tensor.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return (tensor - mean) / std


class RandomRightAngleRotation(object):
    def __init__(self, angles=(0, 90, 180, 270)):
        self.angles = angles

    def __call__(self, image):
        angle = self.angles[torch.randint(low=0, high=len(self.angles), size=(1,)).item()]
        if angle == 0:
            return image
        return transforms.functional.rotate(image, angle)


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(dataset_path("CIFAR"), train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(dataset_path("CIFAR"), train=False, download=True)
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
        train_dataset = datasets.cifar.CIFAR100(dataset_path("CIFAR"), train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(dataset_path("CIFAR"), train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(dataset_path("CIFAR"), train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(dataset_path("CIFAR"), train=False, download=True)
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
        train_dir = dataset_path("ImageNet", "train")
        test_dir = dataset_path("ImageNet", "val")

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
        train_dir = dataset_path("ImageNet-100", "train")
        test_dir = dataset_path("ImageNet-100", "val")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = dataset_path("imagenet-r", "train")
        test_dir = dataset_path("imagenet-r", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = dataset_path("imagenet-a", "train")
        test_dir = dataset_path("imagenet-a", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class CUB(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = dataset_path("cub", "train")
        test_dir = dataset_path("cub", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class objectnet(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = dataset_path("objectnet", "train")
        test_dir = dataset_path("objectnet", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = dataset_path("omnibenchmark", "train")
        test_dir = dataset_path("omnibenchmark", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class vtab(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = dataset_path("vtab-cil", "vtab", "train")
        test_dir = dataset_path("vtab-cil", "vtab", "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class domainnet(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(345).tolist()
    domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = dataset_path("DomainNet")

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


class minidomainnet(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(126).tolist()
    domain_names = ["clipart", "painting", "real", "sketch"]

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = dataset_path("DomainNet")
        image_list_dir = os.path.join(root_dir, "splits_mini")

        self.train_data = []
        self.train_targets = []

        train_image_list_paths = [
            os.path.join(image_list_dir, d + "_" + "train" + ".txt") for d in self.domain_names
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
            os.path.join(image_list_dir, d + "_" + "test" + ".txt") for d in self.domain_names
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
            

class officehome(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(65).tolist()
    domain_names = ["Art", "Clipart", "Product", "Real World"]

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = dataset_path("OfficeHomeDataset_10072016")

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


class office31(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(31).tolist()
    domain_names = ["amazon", "dslr", "webcam"]

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = dataset_path("office31")

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


class officecaltech(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(10).tolist()
    domain_names = ["amazon", "caltech", "dslr", "webcam"]

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = dataset_path("office_caltech_10")

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


class imageclef(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(12).tolist()
    domain_names = ['i', 'p', 'c'] # b?

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = dataset_path("image_CLEF")

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


class digitsdg(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(10).tolist()
    domain_names = ['mnist', 'mnist_m', 'svhn', 'syn']

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = dataset_path("digits_dg")

        self.train_data = []
        self.train_targets = []

        self.test_data = []
        self.test_targets = []

        for domain_id, domain_name in enumerate(self.domain_names):
            domain_img_dir = os.path.join(root_dir, domain_name)
            domain_train_img_dir = os.path.join(domain_img_dir, "train")
            domain_test_img_dir = os.path.join(domain_img_dir, "val")
            domain_train_dset = datasets.ImageFolder(domain_train_img_dir)
            domain_test_dset = datasets.ImageFolder(domain_test_img_dir)
            domain_train_images, domain_train_labels = split_images_labels(domain_train_dset.imgs)
            domain_test_images, domain_test_labels = split_images_labels(domain_test_dset.imgs)

            self.train_data.append(domain_train_images)
            self.train_targets.append(domain_train_labels)
            self.test_data.append(domain_test_images)
            self.test_targets.append(domain_test_labels)


class digitsfive(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(10).tolist()
    domain_names = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = dataset_path("dg5")

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


class core50(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()
    domain_names = [f"s{i}" for i in range(1, 12)]

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = dataset_path("core50", "core50_128x128")

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


class rxrx1_balanced300(iData):
    selected_class_count = 300
    use_path = True
    train_trsf = [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        RandomRightAngleRotation(),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    ]
    common_trsf = [transforms.ToTensor(), SelfStandardize()]
    class_order = np.arange(300).tolist()
    domain_names = ["rxrx1_g0", "rxrx1_g1", "rxrx1_g2", "rxrx1_g3", "rxrx1_g4"]
    domain_groups = [
        ["HEPG2-01", "HEPG2-04", "HEPG2-08", "HEPG2-09", "HUVEC-03", "HUVEC-10", "HUVEC-14", "HUVEC-22", "RPE-01", "RPE-08", "RPE-11"],
        ["HEPG2-03", "HEPG2-10", "HUVEC-02", "HUVEC-09", "HUVEC-13", "HUVEC-15", "HUVEC-17", "RPE-07", "RPE-10", "U2OS-04"],
        ["HEPG2-06", "HEPG2-07", "HUVEC-05", "HUVEC-07", "HUVEC-12", "HUVEC-18", "HUVEC-19", "HUVEC-23", "RPE-03", "U2OS-02"],
        ["HEPG2-02", "HEPG2-05", "HUVEC-04", "HUVEC-06", "HUVEC-11", "HUVEC-20", "RPE-02", "RPE-09", "U2OS-01", "U2OS-05"],
        ["HEPG2-11", "HUVEC-01", "HUVEC-08", "HUVEC-16", "HUVEC-21", "HUVEC-24", "RPE-04", "RPE-05", "RPE-06", "U2OS-03"],
    ]

    def download_data(self):
        root_dir = dataset_path("medical", "rxrx1")
        metadata_path = os.path.join(root_dir, "metadata.csv")
        rows = []
        with open(metadata_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("well_type") != "treatment":
                    continue
                image_path = os.path.join(root_dir, "images", row["experiment"], "Plate" + row["plate"], f"{row['well']}_s{row['site']}.png")
                if os.path.exists(image_path):
                    rows.append((row, image_path))

        domain_to_group = {domain: group_id for group_id, domains in enumerate(self.domain_groups) for domain in domains}
        counts_by_sirna = Counter(row["sirna_id"] for row, _ in rows)
        shared_sirnas = sorted(counts_by_sirna.keys(), key=lambda x: (-counts_by_sirna[x], int(x)))[:self.selected_class_count]
        label_map = {sirna_id: label for label, sirna_id in enumerate(shared_sirnas)}
        self.selected_sirna_ids = shared_sirnas

        grouped = defaultdict(lambda: {"train_data": [], "train_targets": [], "test_data": [], "test_targets": []})
        for row, image_path in rows:
            sirna_id = row["sirna_id"]
            if sirna_id not in label_map or row["experiment"] not in domain_to_group:
                continue
            target = label_map[sirna_id]
            group_id = domain_to_group[row["experiment"]]
            split = row.get("dataset", "train")
            if split == "test":
                grouped[group_id]["test_data"].append(image_path)
                grouped[group_id]["test_targets"].append(target)
            else:
                grouped[group_id]["train_data"].append(image_path)
                grouped[group_id]["train_targets"].append(target)

        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []
        for group_id in range(len(self.domain_names)):
            train_data = np.array(grouped[group_id]["train_data"])
            train_targets = np.array(grouped[group_id]["train_targets"], dtype=np.int64)
            test_data = np.array(grouped[group_id]["test_data"])
            test_targets = np.array(grouped[group_id]["test_targets"], dtype=np.int64)
            if len(test_data) == 0 and len(train_data) > 0:
                train_data, train_targets, test_data, test_targets = split_train_val(train_data, train_targets, val_ratio=0.2, seed=42)
            self.train_data.append(train_data)
            self.train_targets.append(train_targets)
            self.test_data.append(test_data)
            self.test_targets.append(test_targets)


class rxrx1_balanced100(rxrx1_balanced300):
    selected_class_count = 100
    class_order = np.arange(100).tolist()


class rxrx1_balanced50(rxrx1_balanced300):
    selected_class_count = 50
    class_order = np.arange(50).tolist()


class fldr(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ]
    common_trsf = [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    class_order = np.arange(5).tolist()
    domain_names = ["mBRSET", "APTOS", "IDRiD", "SUSTech-SYSU", "DRD"]

    def _label(self, row):
        for key in ["label", "diagnosis", "dr_grade", "level", "target", "class"]:
            if key in row and row[key] != "":
                return int(float(row[key]))
        raise KeyError("Cannot find FL-DR label column in {}".format(row.keys()))

    def _image_name(self, row):
        for key in ["name", "image", "image_id", "img", "filename", "file", "path"]:
            if key in row and row[key] != "":
                return row[key]
        raise KeyError("Cannot find FL-DR image column in {}".format(row.keys()))

    def _read_split(self, domain_dir, split_name):
        csv_path = os.path.join(domain_dir, f"{split_name}_seed0.csv")
        data, targets = [], []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                image_name = self._image_name(row)
                image_path = image_name if os.path.isabs(image_name) else os.path.join(domain_dir, "images", image_name)
                if not os.path.splitext(image_path)[1]:
                    for ext in [".jpg", ".jpeg", ".png"]:
                        if os.path.exists(image_path + ext):
                            image_path = image_path + ext
                            break
                if os.path.exists(image_path):
                    data.append(image_path)
                    targets.append(self._label(row))
        return np.array(data), np.array(targets, dtype=np.int64)

    def download_data(self):
        root_dir = dataset_path("medical", "FL-DR")
        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []
        for domain_name in self.domain_names:
            domain_dir = os.path.join(root_dir, domain_name)
            train_data, train_targets = self._read_split(domain_dir, "train")
            val_data, val_targets = self._read_split(domain_dir, "val")
            test_data, test_targets = self._read_split(domain_dir, "test")
            if len(val_data) > 0:
                train_data = np.concatenate([train_data, val_data])
                train_targets = np.concatenate([train_targets, val_targets])
            self.train_data.append(train_data)
            self.train_targets.append(train_targets)
            self.test_data.append(test_data)
            self.test_targets.append(test_targets)


class camelyon17(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    ]
    test_trsf = [transforms.Resize(224, interpolation=InterpolationMode.BICUBIC)]
    common_trsf = [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    class_order = np.arange(2).tolist()
    domain_names = ["center_0", "center_1", "center_2", "center_3", "center_4"]
    domain_incremental = True

    def download_data(self):
        root_dir = dataset_path("medical", "camelyon17")
        metadata_path = os.path.join(root_dir, "metadata.csv")
        self.train_data = [[] for _ in self.domain_names]
        self.train_targets = [[] for _ in self.domain_names]
        self.test_data = [[] for _ in self.domain_names]
        self.test_targets = [[] for _ in self.domain_names]
        with open(metadata_path, newline="") as f:
            for row in csv.DictReader(f):
                center = int(row["center"])
                patient = row["patient"]
                node = row["node"]
                x_coord = row["x_coord"]
                y_coord = row["y_coord"]
                image_path = os.path.join(root_dir, "patches", f"patient_{patient}_node_{node}", f"patch_patient_{patient}_node_{node}_x_{x_coord}_y_{y_coord}.png")
                if not os.path.exists(image_path):
                    continue
                target = int(row["tumor"])
                split = int(row["split"])
                if split == 0:
                    self.train_data[center].append(image_path)
                    self.train_targets[center].append(target)
                else:
                    self.test_data[center].append(image_path)
                    self.test_targets[center].append(target)
        self.train_data = [np.array(x) for x in self.train_data]
        self.train_targets = [np.array(y, dtype=np.int64) for y in self.train_targets]
        self.test_data = [np.array(x) for x in self.test_data]
        self.test_targets = [np.array(y, dtype=np.int64) for y in self.test_targets]


class midog25(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
    ]
    test_trsf = [transforms.Resize(224, interpolation=InterpolationMode.BICUBIC)]
    common_trsf = [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    class_order = np.arange(2).tolist()
    domain_incremental = True

    def download_data(self):
        root_dir = dataset_path("medical", "MIDOG25")
        image_dir = os.path.join(root_dir, "MIDOG25_Binary_Classification_Train_Set")
        metadata_path = os.path.join(root_dir, "MIDOG25_Atypical_Classification_Train_Set.csv")
        scanner_rows = defaultdict(list)
        with open(metadata_path, newline="") as f:
            for row in csv.DictReader(f):
                image_path = os.path.join(image_dir, row["image_id"])
                if not os.path.exists(image_path):
                    continue
                label = 1 if row["majority"].strip().upper() == "AMF" else 0
                scanner_rows[row["Scanner"]].append((image_path, label))

        self.domain_names = sorted(scanner_rows.keys())
        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []
        for scanner in self.domain_names:
            data = np.array([item[0] for item in scanner_rows[scanner]])
            targets = np.array([item[1] for item in scanner_rows[scanner]], dtype=np.int64)
            train_data, train_targets, test_data, test_targets = split_train_val(data, targets, val_ratio=0.3, seed=42)
            self.train_data.append(train_data)
            self.train_targets.append(train_targets)
            self.test_data.append(test_data)
            self.test_targets.append(test_targets)