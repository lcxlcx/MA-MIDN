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



    #
    # def __getitem__(self, index):
    #     if self.mode == 'train':
    #         random.shuffle(self.samples)
    #
    #     image_dir, label = self.samples[index]
    #     images = glob.glob(image_dir)
    #     # images = glob.glob(image_dir + '/*')
    #
    #     t = transforms.Compose(
    #         [transforms.CenterCrop((1488, 1984))])  # centercropping to 1200 to generate 9 400x400 patches
    #
    #     transformations = transforms.Compose([
    #         transforms.ToTensor()
    #     ])
    #
    #     array = []
    #
    #     for i, image_path in enumerate(images):
    #         # print(image_path)
    #         image = Image.open(image_path)
    #         image = np.array(t(image))
    #         # print(image.shape)
    #         # image = np.array(image)
    #         r, c, _ = image.shape
    #         # print("image.shape",image.shape)
    #         for i in range(0, 124 * 12, 124):
    #             for j in range(0, 124 * 16, 124):
    #                 array.append(transformations(image[i:i + 124, j:j + 124, :]))
    #     #                     array.append(transformations(image[i:i+400, j:j+400, :]).float())
    #     #                     array.append(image[i:i+400, j:j+400, :])
    #     # if (i+400 < r) and (j+400 < c):
    #     # array.append(transformations(image[i:i+400, j:j+400, :]).float())
    #     # array.append(image[i:i+400, j:j+400, :])
    #
    #     array = tuple(array)
    #     # print("################### array ###################")
    #     # print(array)
    #     array = torch.stack(array, 0)
    #
    #     return (array, label)


    def __getitem__(self, index):
        # if self.mode == 'train':
        random.shuffle(self.samples)

        image_dir, label = self.samples[index]
        images = glob.glob(image_dir)
        # images = glob.glob(image_dir + '/*')

        t = transforms.Compose(
            [transforms.CenterCrop((448, 700))
             ])  # centercropping to 1200 to generate 9 400x400 patches

        transformations = transforms.Compose([
            transforms.ToTensor()
        ])

        array = []

        for i, image_path in enumerate(images):
            image = Image.open(image_path)
            image = np.array(t(image))
            # print(image.shape)
            # image = np.array(image)
            r, c, _ = image.shape
            # print("image.shape",image.shape)
            for i in range(0, 28 * 16, 28):
                for j in range(0, 28 * 25, 28):
                    array.append(transformations(image[i:i + 28, j:j + 28, :]))

        array = tuple(array)
        # print("################### array ###################")
        # print(array)
        array = torch.stack(array, 0)

        return (array, label)



