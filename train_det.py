# basic python libraries
import os
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# openCV
import cv2

# xml library for parsing xml files
from xml.etree import ElementTree as et

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# helper libraries
from engine import train_one_epoch, evaluate
# import utils
# import transforms as T

# for image augmentations
# import albumentations as A

# from albumentations.pytorch.transforms import ToTensorV2

from my_utils import FruitImagesDataset,train_model,get_object_detection_model,convert_voc_to_coco

# defining the train and test directory
train_dir = '/work/shared/ngmm/scripts/Beyza_Zayim/datachon/data/train'
test_dir = '/work/shared/ngmm/scripts/Beyza_Zayim/datachon/data/test'

OUTPUT_JSON = "/work/shared/ngmm/scripts/Beyza_Zayim/datachon/output/coco_export/instances_train_coco.json"
CLASS_NAMES = ["SC", "SN"]

convert_voc_to_coco(train_dir, OUTPUT_JSON, CLASS_NAMES)


# init dataset train and test
dataset_train = FruitImagesDataset(train_dir, 224, 224)
dataset_test = FruitImagesDataset(test_dir, 224, 224)

# check dataset
print("DATASET: Fruit-Images-Dataset",
      "\n",
      "\nTRAIN:",
      "\nLenght: {} images".format(dataset_train.__len__()),
      "\n"
      "\nTEST:",
      
      "\nLenght: {} images".format(dataset_test.__len__()))


def collate_fn(batch):
    return tuple(zip(*batch))

# split the dataset in train and validation set
torch.manual_seed(1)
indices = torch.randperm(len(dataset_train)).tolist()

# Original full dataset
full_dataset = dataset_train

# Train/validation split
val_split = 0.2
val_size = int(len(full_dataset) * val_split)
train_dataset = torch.utils.data.Subset(full_dataset, indices[:-val_size])
val_dataset = torch.utils.data.Subset(full_dataset, indices[-val_size:])


# define train data loaders
dataloader_train = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=16,
                                               shuffle=True,
                                               num_workers=2,
                                               collate_fn=collate_fn)

# define validation data loaders
dataloader_val = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=16,
                                             shuffle=False,
                                             num_workers=2,
                                             collate_fn=collate_fn)

# define test data loaders
dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=16,
                                              shuffle=False,
                                              num_workers=2,
                                              collate_fn=collate_fn)

print("DATASET SPLIT:\n",
      "\nTRAIN: {}".format(len(train_dataset)),
      "\nVALIDATION: {}".format(len(val_dataset)),
      "\nTEST: {}".format(len(dataset_test)))


# to train on gpu if selected.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


num_classes = 3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# get the object detection model
model = get_object_detection_model(num_classes)

# move model to the right device
model.to(device)

# define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params,
                            lr=0.005,
                            momentum=0.9,
                            weight_decay=0.0005)

# define learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)




train_accuracies = []
val_accuracies = []





# Example usage:
save_path = "/work/shared/ngmm/scripts/Beyza_Zayim/datachon/output/models"  # Change this to your desired path


# Train the model
train_model(model, dataloader_train, dataloader_val, optimizer, lr_scheduler, device, num_epochs=50, save_path=save_path,patience=10)
