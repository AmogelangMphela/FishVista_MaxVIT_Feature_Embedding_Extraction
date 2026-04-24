import argparse
import os 
import json
import numpy as np
import torch
import torch.nn as nn
from utils import get_mean_std, SquarePadding
from dataloader import create_dataset, get_transforms
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description = "Fishvista training")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--lr_warmup", default=50, type=int, help = "Warm up steps for learning rate")
    parser.add_argument("--model", default = "maxvit_t", type=str, choices=["maxvit_t"], help="model type")
    parser.add_argument("--batch_size", default = 128, type = int, help = "batch size")
    parser.add_argument("--epochs", default=150, type=int, help="number of epochs to train")
    parser.add_argument("--dataset", default="fishvista", type=str)
    parser.add_argument("--optimizer", default="AdamW", type=str, help="optimizer to use")
    parser.add_argument("--decay", default=0.1, type=float, help="weight decay")
    parser.add_argument("--wandb", action="store_true", help="wandb logging")
    parser.add_argument("--seed", default=None, type=int, help="random seed")
    parser.add_argument("--name", default="0", type=str, help="name of run")
    parser.add_argument("--no_augment", dest="augment", action="store_false", help=" deactivate augmentation by default it is activated")
    parser.add_argument("--dataload_type", default="FishVista_normal", type=str)
    return parser.parse_args()

ARGS = parse_args()
if ARGS.seed is not None:
    SEED =  ARGS.seed
else:
    SEED = np.random.randint(10000)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


OPTIMIZER = ARGS.optimizer
DATASET = ARGS.dataset
BATCH_SIZE = ARGS.batch_size
MODEL = ARGS.model
WANDB = ARGS.wandb
LR = ARGS.lr
EPOCHS = ARGS.epochs
START_EPOCH = 0
WARMUP = ARGS.lr_warmup
LOAD_TYPE = ARGS.dataload_type

stats_transforms = get_transforms(224,'stats')

if DATASET == 'fishvista':
    stats_data = create_dataset("Classifiction_dataset/train_standardized.csv", "Classifiction_dataset/Train", stats_transforms)
    mean, std = get_mean_std(stats_data)

    if ARGS.augment:
        train_transforms = get_transforms(224, 'train', mean, std)
        train_dataset = create_dataset("Classifiction_dataset/train_standardized.csv", "Classifiction_dataset/Train", train_transforms)

        val_test_transforms = get_transforms(224, 'val_test', mean, std)
        val_dataset = create_dataset("Classifiction_dataset/val_standardized.csv", "Classifiction_dataset/Val", val_test_transforms)

        embeddings_transforms = get_transforms(224, 'val_test', mean, std)
        embedding_dataset = create_dataset("Classifiction_dataset/train_standardized.csv", "Classifiction_dataset/Train", embeddings_transforms)
else:
    print("Dataset Unavailable")


dataloader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dataloader_val = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dataloader_embeddings = DataLoader(embedding_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


def learning_rate_scheduler(current_epoch):
    if current_epoch < WARMUP:
        return float(current_epoch + 1) / float(WARMUP)
    else:
        progress = float(current_epoch - WARMUP)/float(EPOCHS - WARMUP)
        return 0.5 * (1.0 + math.cos(math.pi * progress))