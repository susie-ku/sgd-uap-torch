import enum
import os
import torch

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset

from functools import partial
torch.manual_seed(42)

@enum.unique
class Datasets(str, enum.Enum):
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    ImageNet = "ImageNet"


def vit_features(image, feature_extractor):
    return feature_extractor(
            images=image, 
            return_tensors="pt"
        )['pixel_values'].squeeze(0)


def get_vit_transforms(feature_extractor):
    return transforms.Compose([
        transforms.ToTensor(),
        partial(vit_features, feature_extractor=feature_extractor)
    ])


def get_dataset(path_to_data, dataset_name, transform=None):
    path = os.path.join(path_to_data, dataset_name)
    if dataset_name == Datasets.CIFAR10:
        dataset = datasets.CIFAR10(path, train=False, download=True, transform=transform)
    elif dataset_name == Datasets.CIFAR100:
        dataset = datasets.CIFAR100(path, train=False, download=True, transform=transform)
    elif dataset_name == Datasets.ImageNet:
        dataset = datasets.ImageFolder(path, transform=transform)
    else:
        raise NotImplementedError()
    
    return dataset

class IndexedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.transform:
            idx, image, label = self.dataset[idx]
            image = self.transform(image)
        else:
            image, label = self.dataset[idx]
        return idx, image, label

class TransformerIndexedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.transform:
            idx, image, label = self.dataset[idx]
            image = self.dataset.transform(image)
        else:
            image, label = self.dataset[idx]
        return idx, image, label

    