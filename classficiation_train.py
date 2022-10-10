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
from torch.autograd import Variable
from omegaconf import OmegaConf as OC 
import os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
from preprocessing import mk_folder

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

class TotalDataset(Dataset):
    def __init__(self, labels=None, transformer=None, mode="train"):
        self.transformer = transformer
        self.mode = mode
        if self.mode == "train" :
            self.labels = labels
        elif self.mode == "val" :
            self.labels = labels
            
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, i):
        img = cv2.imread(self.labels["File"].iloc[i]).astype(np.float32)/255
        img = cv2.resize(img, dsize=(384,384))
        if self.mode == "train":
            if self.transformer != None:
                img = self.transformer(image=img)
                img = np.transpose(img["image"], (2,0,1))
            else:
                img = np.transpose(img, (2,0,1))
            return {
                "img" : torch.tensor(img, dtype=torch.float32),
                "label" : torch.tensor(self.labels["Class"].iloc[i], dtype=torch.long)
            }
        elif self.mode == "val":
            img = np.transpose(img, (2,0,1))
            return {
                "img" : torch.tensor(img, dtype=torch.float32),
                "label" : torch.tensor(self.labels["Class"].iloc[i], dtype=torch.long)
            }
        
        else:
            img = np.transpose(img, (2,0,1))
            return {
                "img" : torch.tensor(img, dtype=torch.float32)
            }
def train_step(batch_item, epoch, batch, training):
    img = batch_item['img'].to(device)
    label = batch_item['label'].to(device)
    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(img)
            loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        pred = torch.argmax(output, axis=1)
        acc = torch.sum(pred == label)
        return loss, acc
    else:
        model.eval()
        with torch.no_grad():
            output = model(img)
            loss = criterion(output, label)
            pred = torch.argmax(output, axis=1)
            acc = torch.sum(pred == label)
            
        return loss, acc

if __name__ == '__main__':
    path = OC.load('path.yaml')
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
    labels = pd.read_csv(path.parent + 'total.csv')
    FOLDS = 5
    skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle=True)
    device = torch.device("cuda:0")
    dropout_rate = 0.1
    class_num = 12
    learning_rate = 1e-4
    BATCH_SIZE = 48
    EPOCHS = 25
    MODELS = 'efficientnet-b7'
    #MODELS = 'regnety_040'
    mk_folder('models')
    save_path = f"{path.weight.class_weight}/Py_{MODELS}_GAN_0922_{EPOCHS}"
    folder_train_idx = 0
    folder_val_idx = 0
    n=0
    
    for train_idx, val_idx in skf.split(range(labels.shape[0]), labels["Class"]):
        n = n + 1

        if (n == 1) or (n == 2) or (n == 3) or (n == 4):
            continue

        albumentations_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(),
            A.ShiftScaleRotate(),
            A.GaussNoise(),
            A.RandomGamma(),
        ])

        model = EfficientNet.from_pretrained(MODELS, advprop=True, num_classes=class_num)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5)
        criterion = nn.CrossEntropyLoss()

        train_dataset = TotalDataset(labels.iloc[train_idx], transformer=albumentations_transform)
        val_dataset = TotalDataset(labels.iloc[val_idx], transformer=albumentations_transform, mode="val")

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=False)

        sample_batch = next(iter(train_dataloader))

        loss_plot, val_loss_plot = [], []

        for epoch in range(EPOCHS):
            total_loss, total_val_loss = 0, 0
            total_acc, total_val_acc = 0, 0

            tqdm_dataset = tqdm(enumerate(train_dataloader))
            training = True
            for batch, batch_item in tqdm_dataset:
                batch_loss, batch_acc = train_step(batch_item, epoch, batch, training)
                total_loss += batch_loss
                total_acc += batch_acc

                tqdm_dataset.set_postfix({
                    'Epoch': epoch + 1,
                    'Loss': '{:06f}'.format(batch_loss.item()),
                    'Total Loss' : '{:06f}'.format(total_loss/(batch+1)),
                    'Total Acc' : '{:06f}'.format((total_acc/((batch+1) * 48)) * 100)
                })
            loss_plot.append(total_loss/(batch+1))

            tqdm_dataset = tqdm(enumerate(val_dataloader))
            training = False
            for batch, batch_item in tqdm_dataset:
                batch_loss, batch_acc = train_step(batch_item, epoch, batch, training)
                total_val_loss += batch_loss
                total_val_acc += batch_acc

                tqdm_dataset.set_postfix({
                    'Epoch': epoch + 1,
                    'Val Loss': '{:06f}'.format(batch_loss.item()),
                    'Total Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                    'Total Val Acc' : '{:06f}'.format((total_val_acc/((batch+1) * 48)) * 100)
                })
            val_loss_plot.append(total_val_loss/(batch+1))
            scheduler.step(total_val_loss/(batch+1))

            if np.min(val_loss_plot) == val_loss_plot[-1]:
                torch.save(model, save_path + f"_{n}.pt")
                print("## Model Save")
