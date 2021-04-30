import os
import glob
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
import random

class PatchMethod(torch.utils.data.Dataset):
    def __init__(self, root='/home/DataSets/', mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.raw_samples = glob.glob(root + '/*/*')
        self.samples = []
        for raw_sample in self.raw_samples:
            self.samples.append((raw_sample, int(raw_sample.split('/')[-2])))


    def __len__(self):
        return len(self.samples)







