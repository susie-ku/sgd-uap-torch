import ast
import enum
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
import torch
# from torch.optim.optimizer import zero_grad

from torchvision.models import (
    # densenet121,
    densenet161,
    efficientnet_b0, 
    efficientnet_b3,
    # inception_v3,
    resnet101,
    resnet152,
    vgg19,
    wide_resnet101_2,
    # wide_resnet50_2
)

from torchvision.models import (
    # DenseNet121_Weights,
    DenseNet161_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    # Inception_V3_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    VGG19_Weights,
    Wide_ResNet101_2_Weights,
    # Wide_ResNet50_2_Weights
)

from flags import parse_handle
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification, AutoFeatureExtractor

from src.attackers import (
    SparseSVAttack,
    PixelSparseSVAttack,
    BlockSparseSVAttack,
    UniversalBlockSparseSVAttack
)

from src.datasets import (
    get_dataset,
    get_vit_transforms,
    Datasets,
    IndexedDataset,
    TransformerIndexedDataset
)

from src.models import (
    AttackViTFeatureExtractor,
    ImageNetAttackImageTransform,
    VictimModel,
    ImageNetModels,
    ImageNetTransformers
)

random.seed(0)
torch.manual_seed(42)


@enum.unique
class Attackers(enum.Enum):
    Sparse = SparseSVAttack
    PixelSparse = PixelSparseSVAttack
    BlockSparse = BlockSparseSVAttack
    UniversalBlockSparse = UniversalBlockSparseSVAttack


parser = parse_handle()
args = parser.parse_args()

assert args.attacker in Attackers.__members__
attacker_class = Attackers.__members__[args.attacker].value
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ImageNetModels = [
    # (DenseNet121_Weights, densenet121),
    (DenseNet161_Weights, densenet161),
    (EfficientNet_B0_Weights, efficientnet_b0), 
    (EfficientNet_B3_Weights, efficientnet_b3),
    # (Inception_V3_Weights, inception_v3),
    (ResNet101_Weights, resnet101),
    (ResNet152_Weights, resnet152),
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


def evaluate_asr():

    df = []

    dataset = IndexedDataset(
        get_dataset(
            '/media/ssd-3t/kkuvshinova/hdd/ImageNet',
            Datasets.ImageNet
        )
    )

    paths_indx = []
    for file in sorted(os.listdir(args.path_to_indx)):
        filename = os.fsdecode(file)
        if 'vit' not in filename and 'deit' not in filename:
            paths_indx.append(os.path.join(args.path_to_indx, filename))
        else:
            continue

    paths_answrs = []
    for file in sorted(os.listdir(args.path_to_answers)):
        filename = os.fsdecode(file)
        if 'vit' not in filename and 'deit' not in filename:
            paths_answrs.append(os.path.join(args.path_to_answers, filename))
        else:
            continue

    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset,
        [args.attack_train_split_num, len(dataset) - args.attack_train_split_num],
        generator=torch.Generator().manual_seed(42)
    )
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    train_dataset_ = IndexedDataset(
        train_dataset, transform=EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
    )
    model_current = str(weights.__class__.__name__).split('_Weights', 1)[0]
    print(model_current)

    # feature_extractor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    # transform = get_vit_transforms(feature_extractor)
    # train_dataset_ = TransformerIndexedDataset(train_dataset, Datasets.ImageNet)
    # train_dataset_.dataset.transform = transform
    # model_current = str('google/vit-base-patch16-224').split('/')[-1]
    # print(model_current)
    # model.eval()

    train_loader = DataLoader(
        dataset=train_dataset_,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=False
    )

    attacked_layer = 'features.denseblock2.denselayer2' # random.choice(layers) # add your layer name-key
    victim = VictimModel(model, attacked_layer)

    if 'Block' not in args.attacker:
        attacker = attacker_class(
            args.p, args.q, args.top_k,
            victim, init_truncation=args.init_truncation,
            seed=args.seed
        )
    else:
        attacker = attacker_class(
            args.p, args.q, args.top_k, args.patch_size,
            victim, init_truncation=args.init_truncation,
            seed=args.seed
        )

    if 'Universal' in args.attacker:
        attack, _ = attacker.train(
            train_loader,
            n_steps=args.n_steps,
            reduction_steps=args.reduction_steps,
            accumulation_steps=args.accumulation_steps
        )
    else:
        attack, _ = attacker.train(
            train_loader,
            n_steps=args.n_steps,
            reduction_steps=args.reduction_steps,
        )

    for i, (weihts_, model_) in enumerate(ImageNetModels):
        model = model_(weights=weihts_.IMAGENET1K_V1)

        weights_eval = ImageNetAttackImageTransform(
            weihts_.IMAGENET1K_V1.transforms(),
            alpha=args.alpha,
            attack=attack,
            model=model_current,
            q=args.q,
            top_k=args.top_k,
            patch_size=args.patch_size,
            layer=attacked_layer
        )
        eval_dataset_ = IndexedDataset(
            eval_dataset,
            transform=weights_eval
        )

        eval_loader = DataLoader(
            dataset=eval_dataset_,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=False
        )

        correct_idxs = set(pd.read_csv(paths_indx[i])['0'])
        wo_attack_prediction=pd.read_csv(paths_answrs[i])
        model.to(device)
        model.eval()

        with torch.no_grad():
            wo_attack_correct_count = 0
            after_attack_correct_count = 0
            asr_count = 0
            fr_count = 0
            count = 0
            print(paths_indx[i])
            print(paths_answrs[i])
            correct_idxs = set(pd.read_csv(paths_indx[i])['0'])
            wo_attack_prediction=pd.read_csv(paths_answrs[i])
            print(str(weihts_.IMAGENET1K_V1.__class__.__name__).split('_Weights', 1)[0])

            for it, (index, batch, label) in enumerate(tqdm(eval_loader)):
                # if it > 3:
                #     break
                batch = batch.cuda()
                index = index.tolist()

                correct_wo_attack = np.array(
                    [idx in correct_idxs for idx in index]
                )

                try:
                    after_attack_prediction = model(batch).argmax(dim=-1).cpu()
                except AttributeError:
                    after_attack_prediction = model(batch)[0].argmax(dim=-1).cpu()
                wo_attack_pred = torch.squeeze(torch.tensor(wo_attack_prediction.set_index('pic_index').iloc[index].values))
                fr = after_attack_prediction != wo_attack_pred
                asr = after_attack_prediction != label
                fr_count += fr.float().sum()

                asr_count += asr[correct_wo_attack].float().sum()
                wo_attack_correct_count += correct_wo_attack.sum()

                correct_after_attack = after_attack_prediction == label
                after_attack_correct_count += correct_after_attack.sum()

                count += batch.shape[0]

            asr = asr_count / wo_attack_correct_count
            wo_attack_acc = wo_attack_correct_count / count
            after_attack_acc = after_attack_correct_count / count
            fr = fr_count / count

            print(
                f'Accuracy for non-attacked model: {wo_attack_acc:.4f}, Accuracy for attacked model: {after_attack_acc:.4f}, ASR unbiased: {asr:.4f}, Attacked Layer: {attacked_layer}, FR: {fr:.4f}')
            lst = [
                'EfficientNet_B0',
                str(weihts_.IMAGENET1K_V1.__class__.__name__).split('_Weights', 1)[0],
                round(wo_attack_acc, 4),
                after_attack_acc.detach().numpy(),
                asr.detach().numpy(),
                attacked_layer,
                fr.detach().numpy(),
                args.q,
                args.patch_size,
                args.alpha,
                args.top_k,
                paths_indx[i],
                paths_answrs[i]
            ]
            df.append(lst)

    paths_indx = []
    for file in sorted(os.listdir(args.path_to_indx)):
        filename = os.fsdecode(file)
        if 'vit' in filename or 'deit' in filename:
            paths_indx.append(os.path.join(args.path_to_indx, filename))
        else:
            continue

    paths_answrs = []
    for file in sorted(os.listdir(args.path_to_answers)):
        filename = os.fsdecode(file)
        if 'vit' in filename or 'deit' in filename:
            paths_answrs.append(os.path.join(args.path_to_answers, filename))
        else:
            continue

    for i, (weihts, model) in enumerate(ImageNetTransformers):
        
        feature_extractor = AttackViTFeatureExtractor(
            alpha=args.alpha,
            attack=attack,
            attack_applied=True,
            model = model_current,
            q=args.q,
            top_k=args.top_k,
            patch_size=args.patch_size,
            layer=attacked_layer
        )
        model = model.from_pretrained(weihts)
        transform = get_vit_transforms(feature_extractor)
        eval_dataset_ = TransformerIndexedDataset(eval_dataset, Datasets.ImageNet)
        eval_dataset_.dataset.transform = transform

        eval_loader = DataLoader(
            dataset=eval_dataset_,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=False
        )

        model.to(device)
        model.eval()

        with torch.no_grad():
            wo_attack_correct_count = 0
            after_attack_correct_count = 0
            asr_count = 0
            fr_count = 0
            count = 0

            print(paths_indx[i])
            print(paths_answrs[i])
            correct_idxs = set(pd.read_csv(paths_indx[i])['0'])
            wo_attack_prediction=pd.read_csv(paths_answrs[i])

            for it, (index, batch, label) in enumerate(tqdm(eval_loader)):
                # if it > 3:
                #     break
                batch = batch.cuda()
                index = index.tolist()
                model = model.to('cuda')

                correct_wo_attack = np.array(
                    [idx in correct_idxs for idx in index]
                )

                after_attack_prediction = victim.predict(batch).cpu()
                wo_attack_pred = torch.squeeze(torch.tensor(wo_attack_prediction.set_index('pic_index').iloc[index].values))
                fr = after_attack_prediction != wo_attack_pred
                asr = after_attack_prediction != label
                fr_count += fr.float().sum()

                asr_count += asr[correct_wo_attack].float().sum()
                wo_attack_correct_count += correct_wo_attack.sum()

                correct_after_attack = after_attack_prediction == label
                after_attack_correct_count += correct_after_attack.sum()

                count += batch.shape[0]

            asr = asr_count / wo_attack_correct_count
            wo_attack_acc = wo_attack_correct_count / count
            after_attack_acc = after_attack_correct_count / count
            fr = fr_count / count

            print(
                f'Accuracy for non-attacked model: {wo_attack_acc:.4f}, Accuracy for attacked model: {after_attack_acc:.4f}, ASR unbiased: {asr:.4f}, Attacked Layer: {attacked_layer}, FR: {fr:.4f}')
            lst = [
                'EfficientNet_B0',
                str(weihts).split('/')[-1],
                round(wo_attack_acc, 4),
                after_attack_acc.detach().numpy(),
                asr.detach().numpy(),
                attacked_layer,
                fr.detach().numpy(),
                args.q,
                args.patch_size,
                args.alpha,
                args.top_k,
                paths_indx[i],
                paths_answrs[i]
            ]
            df.append(lst)
    # df = list(map(list, zip(*df)))
    lst_names = [
        'Attack Model',
        'Eval Model',
        'Accuracy Before Attack',
        'Accuracy After Attack',
        'ASR',
        'Attacker Layer',
        'FR',
        'q',
        'Patch Size',
        'alpha',
        'Top k',
        'Path to Indices',
        'Path To Answers'
    ]
    df = pd.DataFrame(df, columns=lst_names)
    df.to_csv(f'EfficientNet_B0_transfer_wo_trunc_{args.alpha}.csv', index=False)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    evaluate_asr()
