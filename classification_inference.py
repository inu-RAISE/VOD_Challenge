# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from glob import glob

from PIL import Image
import cv2
from tqdm import tqdm

import os
import shutil
import json

import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from efficientnet_pytorch import EfficientNet
from pprint import pprint
import random

import os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from omegaconf import OmegaConf as OC 

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

decoder = {"0" : "car_back",
           "1" : "car_side",
           "2" : "car_front",
           "3" : "truck_back",
           "4" : "truck_side",
           "5" : "truck_front",
           "6" : "motorcycle_back",
           "7" : "motorcycle_side",
           "8" : "motorcycle_front",
           "9" : "bicycle_back",
           "10" : "bicycle_side",
           "11" : "bicycle_front"}

path = OC.load('path.yaml')


v1_txt = open(path.result.v1, "r") # Previously inferred txt file
v2_txt = open(path.result.v2,"r") # Previously inferred txt file

def infer(txt,test_name):
    if test_name == 'v1':
        data_path = path.test.v1
    if test_name == 'v2':
        data_path = path.test.v2
    total_txt = []
    lines = txt.readlines()
    for line in lines:
        line = line.split(" ")
        total_txt.append(line)

    model = [torch.load(f"{path.weight.class_weight}/Py_efficientnet-b7_GAN_0922_25_{i}.pt", map_location="cuda:0") for i in range(1, 6)]

    m = nn.Softmax(dim=1)
    result_txt = []

    for i in tqdm(total_txt):
        file_result = []
        if len(i) == 1:
            file_result.append(i[0].replace("\n", ""))
            result_txt.append(file_result)
            continue
        else:
            file_result.append(i[0])
        img_source = data_path + i[0].replace(".txt", ".jpg")
        img = cv2.imread(img_source).astype(np.float32)/255
        
        for j in range((len(i) - 1) // 6):
            min_x = int(float(i[3 + j*6]))
            min_y = int(float(i[4 + j*6]))
            max_x = int(float(i[5 + j*6]))
            max_y = int(float(i[6 + j*6]))
            
            img_ = img[min_y:max_y, min_x:max_x, :]
            img_ = cv2.resize(img_, (384, 384))
            img_ = np.transpose(img_, (2,0,1))
            img_ = torch.tensor(img_, dtype=torch.float32)
            img_ = img_.unsqueeze(0)
            img_ = img_.to("cuda:0")
            
            proportion = []
            
            for k in model:
                output = k(img_)
                proportion.append(m(output).cpu().detach().numpy())
                
            total_proportion = np.mean(proportion, 0)
            output = str(int(np.argmax(total_proportion)))
            output = decoder[output]
            
            file_result.append(output)
            file_result.append(float(np.max(total_proportion)))
            file_result.append(min_x)
            file_result.append(min_y)
            file_result.append(max_x)
            file_result.append(max_y)
            
        result_txt.append(file_result)
    print(result_txt)
    submission = open(f"./submission/final_{test_name}_submission.txt", "w")
    for i in tqdm(result_txt):
        line = ""
        for j in i:
            line = line + str(j) + " "
        line = line + "\n"
            
        submission.write(line)
    submission.close()


if __name__ == '__main__':
    infer(v1_txt,'v1')
    infer(v2_txt,'v2')
