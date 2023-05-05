import argparse
import numpy as np

def parse_handle():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset: cifar10 or imagenet", type=str, default='imagenet')
    parser.add_argument('--batch_size', help="batch_size", required=False, type=int, default=64)
    parser.add_argument('--patch_size', help="patch_size", required=False, type=int, default=1)
    parser.add_argument('--attacker', help="attacker class", type=str, default='UniversalBlockSparse')
    parser.add_argument('--cifar_checkpoints', help="path to cifar checkpoints", type=str, default='./checkpoints/cifar_best.pth')
    parser.add_argument('--p', help="p", type=float, default=np.inf)
    parser.add_argument('--q', help="q", type=float, default=1)
    parser.add_argument('--top_k', help="top_k", type=int, default=10)
    parser.add_argument('--init_truncation', help="init_truncation", type=float, default=0.9)
    parser.add_argument('--n_steps', help="n_steps", type=int, default=10)
    parser.add_argument('--reduction_steps', help="reduction_steps", required=False, type=int, default=1)
    parser.add_argument('--accumulation_steps', help="accumulation_steps", type=int, default=1)
    parser.add_argument('--seed', help="seed", type=int, default=42)
    parser.add_argument('--path_to_data', help="path to data", type=str, default='./data')
    parser.add_argument('--attack_train_split_num', help="number of items in training set of attack", type=int, default=256)
    parser.add_argument('--alpha', help="attack power", type=float, default=10/255)
    parser.add_argument('--path_to_indx', help="path to save indices for correct answers", type=str, default='./data/correct')
    parser.add_argument('--path_to_answers', help="path to answers of non-attacked model", type=str, default='./data/answers')
    parser.add_argument('--path_to_results', help="path to results for attack", type=str, default='./data/results')
    parser.add_argument('--path_to_images', help="path to resulting images", type=str, default='/media/ssd-3t/kkuvshinova/tpower_images')
    parser.add_argument('--path_to_wandb', help="path to wandb runs metadata", type=str, default='/media/ssd-3t/kkuvshinova/tpower_wandb')

    return parser