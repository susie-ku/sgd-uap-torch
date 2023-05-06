import os
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms._presets import ImageClassification
from torchvision.transforms import functional as F
torch.manual_seed(42)
from PIL import Image

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
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageFeatureExtractionMixin,
    is_torch_tensor
)

from flags import parse_handle

from src.datasets import Datasets

parser = parse_handle()
args = parser.parse_args()

os.makedirs(args.path_to_images, exist_ok=True)

class ImageNetAttackImageTransform(ImageClassification):
    def __init__(
        self,
        transform,
        alpha,
        attack,
        model,
        q,
        top_k,
        patch_size,
        layer
    ):
        super().__init__(
            crop_size=transform.crop_size, 
            resize_size=transform.resize_size, 
            mean=transform.mean,
            std=transform.std, 
            interpolation=transform.interpolation
        )
        self.alpha = alpha
        self.attack = torch.squeeze(attack.cpu())
        self.model = model
        self.q = q
        self.top_k = top_k
        self.patch_size = patch_size
        self.layer = layer

    def forward(self, img):
        from_torch_to_pil = T.ToPILImage()
        img = F.resize(img, self.resize_size[0][0], interpolation=self.interpolation)
        img = F.center_crop(img, self.crop_size[0][0])
        if not isinstance(img, torch.Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        if self.attack.shape[-1] > self.crop_size[0][0]:
            attack = F.center_crop(self.attack, self.crop_size[0][0])
            img += attack * self.alpha
        elif self.attack.shape[-1] < self.crop_size[0][0]:
            try:
                img += self.attack * self.alpha
            except RuntimeError:
                try:
                    attack = torch.nn.functional.pad(input=self.attack, pad=(
                        (self.crop_size[0][0] - self.attack.shape[1]) // 2,
                        (self.crop_size[0][0] - self.attack.shape[1]) // 2,
                        (self.crop_size[0][0] - self.attack.shape[2]) // 2,
                        (self.crop_size[0][0] - self.attack.shape[2]) // 2
                    ), mode='constant', value=0)
                except RuntimeError:
                    attack = torch.nn.functional.pad(input=self.attack, pad=(
                        (self.crop_size[0][0] - self.attack.shape[1]) // 2,
                        (self.crop_size[0][0] - self.attack.shape[1]) // 2 + 1,
                        (self.crop_size[0][0] - self.attack.shape[2]) // 2,
                        (self.crop_size[0][0] - self.attack.shape[2]) // 2 + 1
                    ), mode='constant', value=0)
                img += attack * self.alpha
        else:
            img += self.attack * self.alpha
        
        # attack_ = from_torch_to_pil(self.attack)
        # attack_.save(f"{args.path_to_images}/{self.model}_{self.layer}_attack_{Datasets.ImageNet}_q={self.q}_top-k={self.top_k}_alpha={self.alpha}_patch_size={self.patch_size}.jpeg")
        img = torch.clamp(img, 0., 1.)
        image = T.ToPILImage()(img)
        image.save(f"{self.model}.jpeg")
        # img_a = from_torch_to_pil(img)
        # img_a.save(f"{args.path_to_images}/{self.model}_{self.layer}_img_a_{Datasets.ImageNet}_q={self.q}_top-k={self.top_k}_alpha={self.alpha}_patch_size={self.patch_size}.jpeg")
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

class AttackViTFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):

    model_input_names = ['pixel_values']

    def __init__(
        self,
        alpha,
        attack,
        model,
        q,
        top_k,
        patch_size,
        layer,
        attack_applied=True,
        do_resize=True,
        size=224,
        resample=Image.BILINEAR,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.attack = attack
        self.model = model
        self.q = q
        self.top_k = top_k
        self.patch_size = patch_size
        self.layer = layer
        self.attack_applied = attack_applied if attack_applied is True else False
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def apply_attack(self, image):
        image = T.ToTensor()(image).cuda()
        if self.attack.shape[-1] > self.size:
            attack = F.center_crop(self.attack, self.size)
            image += torch.squeeze(attack) * self.alpha
        else:
            image += torch.squeeze(self.attack) * self.alpha
        image = torch.clamp(image, 0., 1.)
        image = T.ToPILImage()(image)
        image.save(f"{self.model}.jpeg")
        attack_ = T.ToPILImage()(torch.squeeze(self.attack))
        # attack_.save(f"with_trunc_{self.model}_q={self.q}_alpha={self.alpha}.jpeg")
        return image

    def __call__(self, images, return_tensors, **kwargs):

        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (resizing + normalization)
        if self.do_resize and self.size is not None:
            images = [self.resize(image=image, size=self.size, resample=self.resample) for image in images]
        if self.attack_applied:
            images = [self.apply_attack(image=image) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
        

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


class VictimModel(nn.Module):
    def __init__(self, model, layer_name):
        super(VictimModel, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.layers = dict([*self.model.named_modules()])
        assert layer_name in self.layers.keys()
        
        self.model.eval()
        self.layer_outputs = torch.empty(0)

        self.layer = self.layers[self.layer_name]
        
    def layer_outputs_hook(self):
        def hook(_, __, output):
            self.layer_outputs = output
        return hook
    
    def forward(self, x):
        hook = self.layer.register_forward_hook(self.layer_outputs_hook())
        _ = self.model(x)
        hook.remove()
        if isinstance(self.layer_outputs, tuple):
            self.layer_outputs = self.layer_outputs[0]
        return self.layer_outputs

    @torch.no_grad()
    def predict(self, x):
        prediction = self.model(x)
        
        if isinstance(self.model, ViTForImageClassification):
            prediction = prediction.logits
            
        return prediction.argmax(dim=-1).cpu()
        
