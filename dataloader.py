import os
from utils import SquarePadding
import torch
from torchvision import transforms, utils
from skimage import io
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class create_dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.fishes = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.fishes)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        filename = os.path.join(self.root_dir, self.fishes.iloc[idx, 0])
        image = io.imread(filename)
        species_name = self.fishes.iloc[idx, 7]
        species_id = self.fishes.iloc[idx, 17]
        img_name = self.fishes.iloc[idx, 0]
        sample = {'image': image, 'species_name': species_name, 'species_id': species_id, 'img_name': img_name}

        if self.transform:
            tensor_img = transforms.ToTensor()
            ts_img = tensor_img(sample['image'])
            sample['image'] = self.transform(ts_img)

        return sample 
    

def get_transforms(target_size, transform_type, mean = 0, std = 0):

    if transform_type == 'stats':
        transform = transforms.Compose([
            SquarePadding(),
            transforms.Resize((target_size, target_size)),
        ])

    elif transform_type == 'train':
        transform = transforms.Compose([
            SquarePadding(),
            transforms.Resize((target_size, target_size)),
            transforms.CenterCrop((target_size, target_size)),
            transforms.RandomRotation((0,180)),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomAutocontrast(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.Normalize(mean, std)
        ])

    elif transform_type == 'val_test':
        transform = transforms.Compose([
            SquarePadding(),
            transforms.Resize((target_size, target_size)),
            transforms.CenterCrop((target_size, target_size)),
            transforms.Normalize(mean, std),
        ])

    else:
        transform = None
    
    return transform

def show_batch(batch):
    images, species_names, species_ids = batch['image'], batch['species_name'], batch['species_id']
    size = len(images)
    img_size = images.size(2)
    print(species_ids)
    print(images.size(0))
    grid = utils.make_grid(images)
    plt.imshow(grid.numpy().transpose((1,2,0)))

""" for i_batch, sampled_batch in enumerate(dataloader_train):
    print(i_batch, sampled_batch['image'].size())

    if i_batch == 3:
        plt.figure()
        show_batch(sampled_batch)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break   """