import os

import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from utils.toolkit import split_images_labels, split_train_val

MultiDomainDatasets = [
    "domainnet", "minidomainnet", "officehome", "office31", "officecaltech", "imageclef", "digitsdg", "digitsfive", "core50"
]


def use_multi_domain_dataset(dataset_name):
    dataset_name = dataset_name.lower()
    return dataset_name in MultiDomainDatasets


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
        train_dataset = datasets.cifar.CIFAR10("/home/liyuan/data/CIFAR", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("/home/liyuan/data/CIFAR", train=False, download=True)
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
        train_dataset = datasets.cifar.CIFAR100("/home/liyuan/data/CIFAR", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("/home/liyuan/data/CIFAR", train=False, download=True)
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
        train_dataset = datasets.cifar.CIFAR100("/home/liyuan/data/CIFAR", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("/home/liyuan/data/CIFAR", train=False, download=True)
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
        train_dir = "/home/liyuan/data/ImageNet/train/"
        test_dir = "/home/liyuan/data/ImageNet/val/"

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
        train_dir = "/home/liyuan/data/ImageNet-100/train/"
        test_dir = "/home/liyuan/data/ImageNet-100/val/"

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
        train_dir = "./data/imagenet-r/train/"
        test_dir = "./data/imagenet-r/test/"

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
        train_dir = "./data/imagenet-a/train/"
        test_dir = "./data/imagenet-a/test/"

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
        train_dir = "./data/cub/train/"
        test_dir = "./data/cub/test/"

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
        train_dir = "./data/objectnet/train/"
        test_dir = "./data/objectnet/test/"

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
        train_dir = "./data/omnibenchmark/train/"
        test_dir = "./data/omnibenchmark/test/"

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
        train_dir = "./data/vtab-cil/vtab/train/"
        test_dir = "./data/vtab-cil/vtab/test/"

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
        root_dir = "/home/liyuan/data/DomainNet/"

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

    class_order = np.arange(345).tolist()
    domain_names = ["clipart", "painting", "real", "sketch"]

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = "/home/liyuan/data/DomainNet/"
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
        root_dir = "/home/liyuan/data/OfficeHomeDataset_10072016/"

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
        root_dir = "/home/liyuan/data/office31/"

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
        root_dir = "/home/liyuan/data/office_caltech_10/"

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
        root_dir = "/home/liyuan/data/image_CLEF/"

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
        root_dir = "/home/liyuan/data/digits_dg"

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
        root_dir = "/home/liyuan/data/dg5"

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
        root_dir = "/home/liyuan/data/core50/core50_128x128"

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