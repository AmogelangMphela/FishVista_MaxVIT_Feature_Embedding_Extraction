# 🐟 FishVista: Fish Classification & Embedding Extraction

FishVista is a deep learning project for **image-based fish classification** and **feature embedding extraction**. It is designed to support experimentation with classification models while also enabling downstream tasks such as similarity search, clustering, and visualization using learned embeddings.

---

## 📌 Project Overview

This project focuses on:

- Training deep learning models for fish species classification  
- Extracting feature embeddings from trained models  
- Supporting reproducible experiments with configurable pipelines  
- Logging training/debug information for analysis  

---

## 📁 Project Structure

```bash
FishVista/
│
├── Classification_dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── Models/
│
├── wandb/
│
├── config.py
├── dataloader.py
├── extract_embeddings.ipynb
├── tea_debug.log
├── train.log
├── train.py
└── utils.py
