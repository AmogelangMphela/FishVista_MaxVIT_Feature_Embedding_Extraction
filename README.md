# 🐟 FishVista: Fish Classification & Embedding Extraction

FishVista is a deep learning project for **image-based fish classification**. The aim of this git repository is to reproduce the FishVista MaxVIT model results. Then use the trained model to extract feature embeddings. Once the feature embeddings are extracted, they are averaged per species to determine whether the maxVIT misclassifications were due to some fish species having similar features. (The Project is still ongoing as part of my masters research)

---

## 📌 Project Overview

This project focuses on:

- Reproducing the FishVista MaxVIT model 
- Extracting feature embeddings by detaching the MaxVIT classifier head 
- Mapping the embeddings
- Analyzing feature similarities 

---

## 📁 Project Structure

```bash
FishVista/
│
├── Classification_dataset/    #Dataset not uploaded due to GitHub size restrictions
│   ├── train/
│   ├── val/
│   └── test/
│
├── Models/                  # Currently only contains maxVIT model initialization
│
├── wandb/                   #results logged on weights&biases (wandb)
│
├── config.py                #contains Experiment configurations (e.g, learning rate) logged through the CLI
├── dataloader.py            #contains classes and helpers for loading the dataset
├── extract_embeddings.ipynb  #once the model is trained we extract and visualize embeddings for selected species 
├── tea_debug.log
├── train.log                 #log train results (e.g validation loss) per epoch and errors if any occur
├── train.py                  #contains the main training loop (this is where everything comes together almost all defined classes are imported into the training loop)
└── utils.py                  # contains utility functions (e.g calculate std and mean for batchnorm)
