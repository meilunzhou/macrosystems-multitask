# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:53:32 2023

PyTorch dataloader for tree delineation task

@author: zhou.m
"""


import torch
from torch.utils.data import Dataset
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