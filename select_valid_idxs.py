import argparse
import os
import torch

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, ViTForImageClassification
from tqdm import tqdm

from flags import parse_handle
from src.datasets import get_dataset, get_vit_transforms, Datasets, IndexedDataset
from src.models import ImageNetModels, ImageNetTransformers

parser = parse_handle()
args = parser.parse_args()

def evaluate(model, dataset, batch_size):
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False, 
                        drop_last=False, 
                        num_workers=4, 
                        pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    predictions = []
    pred = []
    idxs = []

    with torch.no_grad():
        for i, (index, batch, label) in enumerate(tqdm(loader)):
            prediction = model(batch.to(device))
            
            if isinstance(model, ViTForImageClassification):
                prediction = prediction.logits

            pred_current = prediction.argmax(-1).cpu()
            correct = pred_current == label
            predictions += (correct).data.tolist()
            pred += pred_current.tolist()
            idxs += index.tolist()
    
    predictions = np.array(predictions)
    pred = np.array(pred)
    acc = predictions.mean()
    idxs_correct = predictions.nonzero()[0]
    model.to('cpu')
    return acc, idxs_correct, pred, idxs


if __name__ == '__main__':

    for weihts, model in ImageNetModels:
        weights = weihts.IMAGENET1K_V1
        model = model(weights=weights)
        dataset = IndexedDataset(get_dataset('/image/raid/data/datasets', Datasets.ImageNet, transform=weights.transforms()))
        
        acc, idxs_correct, prediction_wo_attack, idxs = evaluate(model, dataset, args.batch_size)
        file_name = f'{Datasets.ImageNet}_{weights.__class__.__name__}_{acc * 100: .2f}.csv'
        file_pred_name = f'{Datasets.ImageNet}_{weights.__class__.__name__}_preds_wo_attack.csv'
        pd.DataFrame(idxs_correct).to_csv(os.path.join(args.path_to_indx, file_name), index=False)
        pd.DataFrame({
            'pred_wo_attack': prediction_wo_attack,
            'pic_index': idxs
        }).to_csv(os.path.join(args.path_to_answers, file_pred_name), index=False)
    
    for weihts, model in ImageNetTransformers:
        feature_extractor = AutoFeatureExtractor.from_pretrained(weihts)
        model = model.from_pretrained(weihts)

        transform = get_vit_transforms(feature_extractor)
        dataset = IndexedDataset(get_dataset('/image/raid/data/datasets', Datasets.ImageNet, transform=transform))
        acc, idxs_correct, prediction_wo_attack, idxs = evaluate(model, dataset, args.batch_size)
        
        file_name = f"{Datasets.ImageNet}_{weihts.split('/')[-1]}_{acc * 100: .2f}.csv"
        file_pred_name = f"{Datasets.ImageNet}_{weihts.split('/')[-1]}_preds_wo_attack.csv"
        pd.DataFrame(idxs_correct).to_csv(os.path.join(args.path_to_indx, file_name), index=False)
        pd.DataFrame({
            'pred_wo_attack': prediction_wo_attack,
            'pic_index': idxs
        }).to_csv(os.path.join(args.path_to_answers, file_pred_name), index=False)

