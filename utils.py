import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils
from skimage import io
import pandas as pd
import torchvision.transforms.functional as F



class SquarePadding(object):
    def __call__(self, image):
        _, height, width = image.shape

        max_side = max(width, height)
        top = (max_side - height) // 2
        bottom = (max_side - height) - top
        left = (max_side - width) // 2
        right = (max_side - width) - left

        pad_image = F.pad(image, (left, top, right, bottom), 0 , 'edge')

        return pad_image
    
def get_mean_std(dataset, batch_size=128, num_workers=0):
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    channels_sum = 0.
    channels_sq = 0.
    num_batches = 0

    for i_batch , sampled_batch in enumerate(dataLoader):
        channels_sum += torch.mean(sampled_batch['image'], dim = [0, 2, 3])
        channels_sq += torch.mean((sampled_batch['image'])**2, dim = [0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sq / num_batches - mean ** 2) ** 0.5
    print(mean)
    print(std)

    return (mean, std)

