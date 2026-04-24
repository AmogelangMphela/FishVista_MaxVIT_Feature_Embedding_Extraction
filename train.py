import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as F
import torch.optim as optim
from torchvision.models import MaxVit_T_Weights
import torchvision.models as models
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from config import *
from Models.model import custom_model
import wandb


run =  wandb.init(
    entity = "amo4-rhodes-university",
    project = "FishVista_Project",
    config = {
        "model" : MODEL,
        "optimizer": OPTIMIZER,
        "learning_rate" : LR,
        "lr_warmup" : WARMUP,
        "dataload_type" : LOAD_TYPE,
        "dataset" : DATASET,
        "epochs" : EPOCHS,
        "decay" : ARGS.decay,
        "seed" : ARGS.seed
    }
)

maxVIT = custom_model('maxvit_t', 1758)
maxVIT = maxVIT.to(device)
print(maxVIT.classifier)


loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = optim.AdamW(maxVIT.parameters(), lr = 3e-4, weight_decay=0.1)

    
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_scheduler)

def one_epoch_training(epoch_index):
    running_loss = 0.
    total_samples = 0
    accurate_preds = 0

    train_data = []

    for i_batch, batch in enumerate(dataloader_train):
        images, labels = batch['image'].to(device), batch['species_id'].to(device)
        img_names, species_names, labels2 = batch['img_name'], batch['species_name'], batch['species_id']
        optimizer.zero_grad()

        outputs = maxVIT(images)
        preds = outputs.argmax(dim=1)
        accurate_preds += (preds == labels).sum().item()

        losses = loss_fn(outputs, labels)
        loss = losses.mean()
        loss.backward()
        optimizer.step()

        losses_cpu = losses.cpu()
        preds_cpu = preds.cpu()
        
        for i, (img_name, species_name, label) in enumerate(zip(img_names, species_names, labels2)):
            train_data.append({
                "image_name": img_name,
                "Species" : species_name,
                "Prediction": preds_cpu[i],
                "Actual_class": label,
                "Loss" : losses_cpu[i].item()
            })
        
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    scheduler.step()
    
    train_df = pd.DataFrame(train_data)
    train_df.to_csv(f"TrainLoss_per_Image/epoch_{epoch}_Train_data.csv", index=False)

    train_loss = round(running_loss/total_samples, 4)
    train_accuracy = round(accurate_preds/total_samples, 4)
    
    return (train_loss, train_accuracy)


for epoch in range(EPOCHS):
    val_data = []
    best_data = [0]

    maxVIT.train()
    train_loss, train_accuracy = one_epoch_training(epoch)

    run.log({"Loss/train": train_loss, "Accuracy/train": train_accuracy})

    running_vloss = 0.0
    total_samples = 0
    accurate_preds = 0

    maxVIT.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader_val):
            images, labels = batch['image'].to(device), batch['species_id'].to(device)
            img_names, species_names, labels2 = batch['img_name'], batch['species_name'], batch['species_id']

            outputs = maxVIT(images)
            preds = outputs.argmax(dim=1)
            accurate_preds += (preds == labels).sum().item()

            losses = loss_fn(outputs, labels)
            loss = losses.mean()

            losses_cpu = losses.cpu()
            preds_cpu = preds.cpu()

            for i, (img_name, species_name, label) in enumerate(zip(img_names, species_names, labels2)):
                val_data.append({
                    "image_name": img_name,
                    "Species" : species_name,
                    "Prediction": preds_cpu[i],
                    "Actual_class": label,
                    "Loss" : losses_cpu[i].item()
                })

            batch_size = images.size(0)
            running_vloss += loss.item() * batch_size
            total_samples += batch_size

    results_df = pd.DataFrame(val_data)
    results_df.to_csv(f"ValidationLoss_per_Image/epoch_{epoch}_Validation_data.csv", index=False)

    val_loss = round(running_vloss / total_samples, 4)
    val_accuracy = round(accurate_preds / total_samples, 4)

    run.log({"Loss/Val": val_loss, "Accuracy/Val": val_accuracy})
    
    print("Epoch: ", epoch, "/", EPOCHS, " Training loss: ", train_loss, " Train Accuracy: " , train_accuracy ," Validation loss: ", val_loss, "Validation Accuracy: ", val_accuracy)

    if val_accuracy > max(best_data):
        run.log({"Best_Accuracy": val_accuracy})
        torch.save(maxVIT.state_dict(), f"checkpoints/maxvit_best.pth")
        best_data.append(val_accuracy)

    if (epoch + 1) % 10 == 0:
        torch.save(maxVIT.state_dict(), f"checkpoints/maxvit_epoch_{epoch+1}.pth")
        