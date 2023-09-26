# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:53:32 2023

PyTorch dataloader for tree delineation task

@author: zhou.m
"""


import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import os 
import rasterio as rio
import ast

class TreeBoundingBoxes(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        
        self.grouped_data = self.dataframe.groupby('image_path').agg(
            {'xmin': lambda x: x.tolist(),
             'ymin': lambda x: x.tolist(),
             'xmax': lambda x: x.tolist(),
             'ymax': lambda x: x.tolist(),
             'label': lambda x: x.tolist()})
        
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        #return len(self.dataframe)
        return len(self.grouped_data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.grouped_data.index[idx]).replace('\\', '/')
                                                                                      
        try:

            with rio.open(img_path) as dataset:
                image = dataset.read()
                #image = image.reshape(400, 400, 3)

            # Get the bounding box lists
            xmin = ast.literal_eval(self.grouped_data.iloc[idx, 0][0])
            ymin = ast.literal_eval(self.grouped_data.iloc[idx, 1][0])
            xmax = ast.literal_eval(self.grouped_data.iloc[idx, 2][0])
            ymax = ast.literal_eval(self.grouped_data.iloc[idx, 3][0])
            labels = ast.literal_eval(self.grouped_data.iloc[idx, 4][0])

            #print(ast.literal_eval(xmin[0]))

            # Combine into a list of bounding boxes
            boxes = [torch.tensor((x1, y1, x2, y2)) for x1, y1, x2, y2 in zip(xmin, ymin, xmax, ymax)]


            if self.transform:
                image = self.transform(image)

            return image, {
                "boxes": boxes,
                "labels": labels,
            }
        
        except Exception as e:
             # Print an error message and return None for this sample
            print(f"Error loading image at path {img_path}: {str(e)}")
            return None
        
        
        
class TreeClassification(Dataset):
    
    def __init__(self, dataframe, image_dir, height, width, bands_to_use):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.height = height
        self.width = width
        self.bands_to_use = bands_to_use
        self.transform = transforms.Compose([
            transforms.ToTensor()  # Convert to tensor
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path =  str(os.path.join(self.image_dir, self.dataframe.iloc[idx]['plotID'].replace('\\', '/')))+ '_2018_hyperspectral.tif'
        pixel_x = int(self.dataframe.iloc[idx]['pixel_x'])
        pixel_y = int(self.dataframe.iloc[idx]['pixel_y'])
        taxon_id = self.dataframe.iloc[idx]['taxonID']

        # Open the image using rasterio
        with rio.open(image_path) as src:
            
            # Read the image as a numpy array
            img = src.read(out_shape=(len(src.indexes), src.height, src.width))
            #print(img.shape)
            
        # Select the specified bands
        img = img[self.bands_to_use]

        # Calculate cropping bounds
        half_height = self.height // 2
        half_width = self.width // 2
        y_start = max(0, pixel_y - half_height)
        y_end = min(img.shape[1], pixel_y + half_height + 1)
        x_start = max(0, pixel_x - half_width)
        x_end = min(img.shape[2], pixel_x + half_width + 1)

        # Crop the image
        cropped_image = img[:, y_start:y_end, x_start:x_end]

        # Pad the image if needed to match the desired height and width
        cropped_image = self.pad_image(cropped_image, self.height, self.width)

        # Apply transformations
        cropped_image = self.transform(cropped_image)

        return cropped_image, taxon_id

    def pad_image(self, image, target_height, target_width):
        _, height, width = image.shape

        # Calculate padding
        pad_height = max(0, target_height - height)
        pad_width = max(0, target_width - width)

        # Pad the image
        padded_image = np.pad(image, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant')

        return padded_image
