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

    paths_layers = []
    for file in sorted(os.listdir('data/layers/')):
        filename = os.fsdecode(file)
        if not filename.startswith('vit') and not filename.startswith('deit'):
            paths_layers.append(os.path.join('data/layers/', filename))
        else:
            continue

    dataset = IndexedDataset(
        get_dataset(
            '/media/ssd-3t/kkuvshinova/hdd/ImageNet',
            Datasets.ImageNet
        )
    )
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset,
        [args.attack_train_split_num, len(dataset) - args.attack_train_split_num],
        generator=torch.Generator().manual_seed(42)
    )

    attacked_layers = [
        'features.denseblock2.denselayer2',
        'features.2.1.block',
        'features.1.0.block',
        'maxpool2',
        'layer2.3',
        'layer2.3',
        'features.9',
        'layer3.1'
    ]
    top_k = [2509, 2509, 4500, 4471, 157, 157, 157, 157]
    patch_size = [1, 1, 1, 1, 4, 4, 4, 4]

    for i, (weihts, model) in enumerate(ImageNetModels):
        fr_list = []
        q_list = []
        weights = weihts.IMAGENET1K_V1
        model = model(weights=weihts)
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
        victim = VictimModel(model, attacked_layers[i])
        train_objective = []
        test_objective = []
        for q in [0.1, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

            if 'Block' not in args.attacker:
                attacker = attacker_class(
                    args.p, q, top_k[i],
                    victim, init_truncation=args.init_truncation,
                    seed=args.seed
                )
            else:
                attacker = attacker_class(
                    args.p, q, top_k[i], patch_size[i],
                    victim, init_truncation=args.init_truncation,
                    seed=args.seed
                )

            if 'Universal' in args.attacker:
                attack, objective = attacker.train(
                    train_loader,
                    n_steps=args.n_steps,
                    reduction_steps=args.reduction_steps,
                    accumulation_steps=args.accumulation_steps
                )
            else:
                attack, objective = attacker.train(
                    train_loader,
                    n_steps=args.n_steps,
                    reduction_steps=args.reduction_steps,
                )

            train_objective.append(objective)

            weights_eval = ImageNetAttackImageTransform(
                weihts.IMAGENET1K_V1.transforms(),
                alpha=args.alpha,
                attack=attack,
                model=model_current,
                q=q,
                top_k=top_k[i],
                patch_size=patch_size[i],
                layer=attacked_layers[i]
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

                objective = 0.

                for it, (index, batch, label) in enumerate(tqdm(eval_loader)):
                    # if it > 1:
                    #     break
                    batch = batch.cuda()
                    index = index.tolist()

                    correct_wo_attack = np.array(
                        [idx in correct_idxs for idx in index]
                    )

                    after_attack_prediction = model(batch).argmax(dim=-1).cpu()
                    wo_attack_pred = torch.squeeze(torch.tensor(wo_attack_prediction.set_index('pic_index').iloc[index].values))
                    fr = after_attack_prediction != wo_attack_pred
                    asr = after_attack_prediction != label
                    fr_count += fr.float().sum()

                    asr_count += asr[correct_wo_attack].float().sum()
                    wo_attack_correct_count += correct_wo_attack.sum()

                    correct_after_attack = after_attack_prediction == label
                    after_attack_correct_count += correct_after_attack.sum()

                    count += batch.shape[0]
                    batch_objective = attacker.compute_objective(batch)
                    objective += batch_objective

                asr = asr_count / wo_attack_correct_count
                wo_attack_acc = wo_attack_correct_count / count
                after_attack_acc = after_attack_correct_count / count
                fr = (fr_count / count).detach().numpy()
                test_objective.append(objective.cpu().numpy() / len(eval_loader.dataset))

                print(
                    f'Accuracy for non-attacked model: {wo_attack_acc:.4f}, Accuracy for attacked model: {after_attack_acc:.4f}, ASR unbiased: {asr:.4f}, Attacked Layer: {attacked_layers[i]}, FR: {fr:.4f}, q: {q}')
                fr_list.append(fr)
                q_list.append(q)

        plt.plot(q_list, fr_list)
        plt.title(rf'Dependence of the FR on the $q$ on {model_current}', fontsize=10)
        plt.xlabel(r'$q$')
        plt.ylabel('FR')
        plt.savefig(f'FR_q_{model_current}.png')
        plt.close()

        plt.plot(q_list, train_objective, label='Train Objective')
        plt.plot(q_list, test_objective, label='Test Objective')
        plt.title(rf'Dependence of the train and test objectives on the $q$ on {model_current}', fontsize=10)
        plt.xlabel(r'$q$')
        plt.ylabel('Objective')
        plt.savefig(f'Objective_q_{model_current}.png')
        plt.close()
            
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

    paths_layers = []
    for file in sorted(os.listdir('data/layers/')):
        filename = os.fsdecode(file)
        if filename.startswith('vit') or filename.startswith('deit'):
            paths_layers.append(os.path.join('data/layers/', filename))
        else:
            continue

    # multiprocessing.set_start_method('spawn')
            
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

        top_k = 2509
        patch_size = 1
        attacked_layer = 'vit.encoder.layer.0'
        victim = VictimModel(model, attacked_layer)

        fr_list = []
        q_list = []
        train_objective = []
        test_objective = []
        for q in [0.1, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

            if 'Block' not in args.attacker:
                attacker = attacker_class(
                    args.p, q, top_k,
                    victim, init_truncation=args.init_truncation,
                    seed=args.seed
                )
            else:
                attacker = attacker_class(
                    args.p, q, top_k, patch_size,
                    victim, init_truncation=args.init_truncation,
                    seed=args.seed
                )

            if 'Universal' in args.attacker:
                attack, objective = attacker.train(
                    train_loader,
                    n_steps=args.n_steps,
                    reduction_steps=args.reduction_steps,
                    accumulation_steps=args.accumulation_steps
                )
            else:
                attack, objective = attacker.train(
                    train_loader,
                    n_steps=args.n_steps,
                    reduction_steps=args.reduction_steps,
                )

            train_objective.append(objective)

            feature_extractor = AttackViTFeatureExtractor(
                alpha=args.alpha,
                attack=attack,
                attack_applied=True,
                model = model_current,
                q=q,
                top_k=top_k,
                patch_size=patch_size,
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
                objective = 0.

                for it, (index, batch, label) in enumerate(tqdm(eval_loader)):
                    # if it > 1:
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
                    batch_objective = attacker.compute_objective(batch)
                    objective += batch_objective

                asr = asr_count / wo_attack_correct_count
                wo_attack_acc = wo_attack_correct_count / count
                after_attack_acc = after_attack_correct_count / count
                fr = fr_count / count
                test_objective.append(objective.cpu().numpy() / len(eval_loader.dataset))

                print(
                    f'Accuracy for non-attacked model: {wo_attack_acc:.4f}, Accuracy for attacked model: {after_attack_acc:.4f}, ASR unbiased: {asr:.4f}, Attacked Layer: {attacked_layers[i]}, FR: {fr:.4f}, q: {q}')
                fr_list.append(fr)
                q_list.append(q)

        plt.plot(q_list, fr_list)
        plt.title(rf'Dependence of the FR on the $q$ on {model_current}', fontsize=10)
        plt.xlabel(r'$q$')
        plt.ylabel('FR')
        plt.savefig(f'FR_q_{model_current}.png')
        plt.close()

        plt.plot(q_list, train_objective, label='Train Objective')
        plt.plot(q_list, test_objective.cpu(), label='Test Objective')
        plt.title(rf'Dependence of the train and test objectives on the $q$ on {model_current}', fontsize=10)
        plt.xlabel(r'$q$')
        plt.ylabel('Objective')
        plt.savefig(f'Objective_q_{model_current}.png')
        plt.close()


if __name__ == '__main__':
    evaluate_asr()
