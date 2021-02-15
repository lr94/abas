import os

import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder


def get_dataset_by_name(dataset_name: str, root='data') -> (Dataset, Dataset):
    def create_picture_transform_chain(data_augmentation=True,
                                       mean=(0.485, 0.456, 0.406),
                                       std=(0.229, 0.224, 0.225),
                                       size=256,
                                       crop_size=224):
        return tr.Compose((
            tr.Resize(size),

            tr.RandomResizedCrop(crop_size) if data_augmentation else tr.CenterCrop(crop_size),
            *((tr.RandomHorizontalFlip(),) if data_augmentation else ()),

            tr.ToTensor(),

            tr.Normalize(mean, std)
        ))

    transform_train = create_picture_transform_chain(data_augmentation=True)
    transform_test = create_picture_transform_chain(data_augmentation=False)

    # Office31
    if dataset_name in ('amazon', 'dslr', 'webcam'):
        # For Office-31 we use all the source labeled data and all the target unlabeled data
        # No official splits
        office_train = ImageFolder(root=os.path.join(root, 'Office31', dataset_name, 'images'),
                                   transform=transform_train)
        office_test = ImageFolder(root=os.path.join(root, 'Office31', dataset_name, 'images'),
                                  transform=transform_test)

        return office_train, office_test

    # PACS
    if dataset_name in ('art-pacs', 'cartoon', 'photo', 'sketch-pacs'):
        if dataset_name == 'sketch-pacs':
            # Folder name
            dataset_name = 'sketch'
        elif dataset_name == 'art-pacs':
            dataset_name = 'art_painting'

        # For PACS we use all the source labeled data and all the target unlabeled data
        # No official splits
        pacs_train = ImageFolder(root=os.path.join(root, 'PACS', dataset_name),
                                 transform=transform_train)
        pacs_test = ImageFolder(root=os.path.join(root, 'PACS', dataset_name),
                                transform=transform_test)

        return pacs_train, pacs_test

    # Office-Home
    if dataset_name in ('art-oh', 'clipart-oh', 'realworld', 'product'):
        dataset_name = {
            'art-oh': 'Art',
            'clipart-oh': 'Clipart',
            'realworld': 'Real World',
            'product': 'Product'
        }[dataset_name]

        oh_train = ImageFolder(root=os.path.join(root, 'OfficeHome', dataset_name),
                               transform=transform_train)
        oh_test = ImageFolder(root=os.path.join(root, 'OfficeHome', dataset_name),
                              transform=transform_test)

        return oh_train, oh_test

    return None, None


def get_class_count(dataset: Dataset) -> int:
    if isinstance(dataset, ImageFolder):
        return len(dataset.classes)

    raise ValueError("Unknown dataset " + dataset.__class__.__name__)


def prepare_datasets(source: str, target: str, root: str = 'data') -> (Dataset, Dataset, Dataset, Dataset, int):
    train_src, test_src, train_trg, test_trg = *get_dataset_by_name(source, root), *get_dataset_by_name(target, root)

    source_classes = get_class_count(train_src)
    target_classes = get_class_count(train_trg)
    assert source_classes == target_classes

    return train_src, test_src, train_trg, test_trg, source_classes
