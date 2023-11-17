import ast
from collections.abc import MutableMapping
import enum
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import torch
# from torch.optim.optimizer import zero_grad

from flags import parse_handle
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from attacks import uap_sgd

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

parser = parse_handle()
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def nested_children(m: torch.nn.Module):
#     children = dict(m.named_children())
#     output = {}
#     if children == {}:
#         return m
#     else:
#         for name, child in children.items():
#             try:
#                 output[name] = nested_children(child)
#             except TypeError:
#                 output[name] = nested_children(child)
#     return output

# def flatten(d, parent_key='', sep='.'):
#     items = []
#     for k, v in d.items():
#         new_key = parent_key + sep + k if parent_key else k
#         if isinstance(v, MutableMapping):
#             items.extend(flatten(v, new_key, sep=sep).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)

def evaluate_asr():

    dataset = IndexedDataset(
        get_dataset(
            args.path_to_imagenet,
            Datasets.ImageNet
        )
    )
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset,
        [args.attack_train_split_num, len(dataset) - args.attack_train_split_num],
        generator=torch.Generator().manual_seed(42)
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

    attacked_layers = [
        'features.denseblock2.denselayer2',
        'features.2.1.block',
        'features.1.0.block',
        'maxpool2',
        'layer2.3',
        'layer2.3',
        'layer3.1'
    ]

    for i, (weihts, model) in enumerate(ImageNetModels):
        weights = weihts.IMAGENET1K_V1
        model_eval = model(weights=weihts)
        model_train = model(weights=weihts)
        train_dataset_ = IndexedDataset(
            train_dataset, transform=weihts.IMAGENET1K_V1.transforms()
        )

        train_loader = DataLoader(
            dataset=train_dataset_,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=False
        )

        model_current = str(weights.__class__.__name__).split('_Weights', 1)[0]
        print(model_current)

        attack, _ = uap_sgd(
            model_train,
            train_loader,
            nb_epoch=10,
            eps=1,
            beta=12,
            step_decay=0.8,
            y_target=None,
            loss_fn=None, 
            layer_name=None, 
            uap_init=None
        )
        print(attack)
        print(torch.count_nonzero(attack))

        weights_eval = ImageNetAttackImageTransform(
            weihts.IMAGENET1K_V1.transforms(),
            alpha=10/255,
            attack=attack,
            model=model_current,
            q=args.q,
            top_k=args.top_k,
            patch_size=args.patch_size,
            layer='sgd_uap'
        )
        eval_dataset_ = IndexedDataset(
            eval_dataset,
            transform=weights_eval
        )

        eval_loader = DataLoader(
            dataset=eval_dataset_,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=4,
            pin_memory=False
        )

        model_eval.eval()
        model_eval.to('cuda')

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
                # if it > 1:
                #     break
                batch = batch.cuda()
                index = index.tolist()

                correct_wo_attack = np.array(
                    [idx in correct_idxs for idx in index]
                )

                after_attack_prediction = model_eval(batch).argmax(dim=-1).cpu()
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
            fr = (fr_count / count).detach().numpy()

            print(
                f'Accuracy for non-attacked model: {wo_attack_acc:.4f}, Accuracy for attacked model: {after_attack_acc:.4f}, ASR unbiased: {asr:.4f}, FR: {fr:.4f}')
            
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
        feature_extractor = AutoFeatureExtractor.from_pretrained(weihts)
        model = model.from_pretrained(weihts)
        transform = get_vit_transforms(feature_extractor)
        train_dataset_ = TransformerIndexedDataset(train_dataset, Datasets.ImageNet)
        train_dataset_.dataset.transform = transform
        model_current = str(weihts).split('/')[-1]
        print(model_current)

        train_loader = DataLoader(
            dataset=train_dataset_,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=False
        )
        
        attack, _ = uap_sgd(
            model,
            train_loader,
            nb_epoch=10,
            eps=10/255,
            beta=12,
            step_decay=0.8,
            y_target=None,
            loss_fn=None, 
            layer_name=None, 
            uap_init=None
        )
        print(attack)

        feature_extractor = AttackViTFeatureExtractor(
            alpha=10/255,
            attack=attack,
            attack_applied=True,
            model = model_current,
            q=args.q,
            top_k=args.top_k,
            patch_size=args.patch_size,
            layer='sgd_uap'
        )
        model = model.from_pretrained(weihts)
        transform = get_vit_transforms(feature_extractor)
        eval_dataset_ = TransformerIndexedDataset(eval_dataset, Datasets.ImageNet)
        eval_dataset_.dataset.transform = transform

        eval_loader = DataLoader(
            dataset=eval_dataset_,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=4,
            pin_memory=False
        )

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
                # if it > 1:
                #     break
                batch = batch.cuda()
                index = index.tolist()
                model = model.to('cuda')

                correct_wo_attack = np.array(
                    [idx in correct_idxs for idx in index]
                )

                after_attack_prediction = model(batch).logits.argmax(dim=-1).cpu()
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
                f'Accuracy for non-attacked model: {wo_attack_acc:.4f}, Accuracy for attacked model: {after_attack_acc:.4f}, ASR unbiased: {asr:.4f}, FR: {fr:.4f}')



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    evaluate_asr()
