import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms

from torchvision.models import (
    # densenet121,
    densenet161,
    efficientnet_b0, 
    efficientnet_b3,
    inception_v3,
    resnet101,
    resnet152,
    # resnet50,
    vgg19,
    wide_resnet101_2,
    # wide_resnet50_2
)

from torchvision.models import (
    # DenseNet121_Weights,
    DenseNet161_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    Inception_V3_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    # ResNet50_Weights,
    VGG19_Weights,
    Wide_ResNet101_2_Weights,
    # Wide_ResNet50_2_Weights
)

from transformers import BatchFeature, FeatureExtractionMixin
from transformers import ViTForImageClassification

sys.path.append(os.path.realpath('..'))

from attacks import uap_sgd
from utils import loader_imgnet, model_imgnet, evaluate

dir_data = '/media/ssd-3t/kkuvshinova/hdd/ImageNet'

# load model
ImageNetModels = [
    # (DenseNet121_Weights, densenet121),
    (DenseNet161_Weights, densenet161),
    (EfficientNet_B0_Weights, efficientnet_b0), 
    (EfficientNet_B3_Weights, efficientnet_b3),
    (Inception_V3_Weights, inception_v3),
    (ResNet101_Weights, resnet101),
    (ResNet152_Weights, resnet152),
    # (ResNet50_Weights, resnet50),
    (VGG19_Weights, vgg19),
    (Wide_ResNet101_2_Weights, wide_resnet101_2),
    # (Wide_ResNet50_2_Weights, wide_resnet50_2)
]


ImageNetTransformers = [
    ('facebook/deit-base-patch16-224', ViTForImageClassification),
    # ('facebook/deit-small-patch16-224', ViTForImageClassification),
    ('google/vit-base-patch16-224', ViTForImageClassification),
    # ('WinKawaks/vit-small-patch16-224', ViTForImageClassification)
]

nb_epoch = 10
eps = 10 / 255
beta = 12
step_decay = 0.7
for weights, model in ImageNetModels:
    model = model(weights=weights)
    dataset = datasets.ImageFolder('/media/ssd-3t/kkuvshinova/hdd/ImageNet', transform=weights.IMAGENET1K_V1.transforms())
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset,
        [256, len(dataset) - 256],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=False
    )
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=False
    )
    uap, losses = uap_sgd(model, train_loader, nb_epoch, eps, beta, step_decay)
    _, _, _, _, outputs, labels = evaluate(model, eval_loader, uap = uap)
    print('Accuracy:', sum(outputs == labels) / len(labels))
