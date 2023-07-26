import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import wandb
import argparse
import dataset 
from torchvision.transforms.transforms import Compose, ToTensor, Resize, CenterCrop, RandomHorizontalFlip, RandomRotation, Normalize
import pandas as pd
import argparse
import random

parser = argparse.ArgumentParser()

parser.add_argument('--split', type=int)  # name of the folder where model will be saved (same name for results)
parser.add_argument('--gpu', type=str)

args = parser.parse_args()
split = args.split

seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = resnet.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
resnet.fc = nn.Linear(num_ftrs, 39)

# Load state dict 
state_dict = torch.load('/home/ubuntu/local/Resnet50/models/weighted/noun_split'+str(split)+'.pt')


# Load state dict into the model
resnet.load_state_dict(state_dict)

resnet.to(device)

# Set the model to evaluation mode
resnet.eval()

print("DEVICE = ", device)


#############################################################################################################################

transforms = Compose([Resize(224),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

results = pd.DataFrame(columns=['frame', 'pred', 'gt'])
# annots_train = pd.read_csv("/home/ubuntu/local/Resnet50/csv/noun/split"+str(split)+"/train.csv")
# image_train = sorted(annots_train["frame_path"].unique().tolist())
correct = 0 
preds = []
labels = []

# for img in image_train:
#     image = Image.open("/home/ubuntu/local/Resnet50/GTEA/train/"+img)
#     image = transforms(image).to(device)
#     image = image.unsqueeze(0)  # Add batch dimension
#     output = resnet(image)
#     label = annots_train.loc[annots_train["frame_path"]==img].label.item()
#     labels.append(label)
#     _, pred = torch.max(output,1)
#     preds.append(pred.item())
#     new_row = {'frame' : img, 'pred' : pred.item(), 'gt' : label}
#     results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

# count1 = sum(x == y for x, y in zip(preds, labels))
# print("ACCURACY for val = " ,count1/24601*100)

annots_val = pd.read_csv("/home/ubuntu/local/Resnet50/csv/noun/split"+str(split)+"/test.csv")
image_val = sorted(annots_val["frame_path"].unique().tolist())
print(len(image_val))
for img in image_val:
    image = Image.open("/home/ubuntu/local/Resnet50/GTEA/train/"+img)
    image = transforms(image).to(device)
    image = image.unsqueeze(0)  # Add batch dimension
    output = resnet(image)
    label = annots_val.loc[annots_val["frame_path"]==img].label.item()
    labels.append(label)
    _, pred = torch.max(output,1)
    preds.append(pred.item())
    new_row = {'frame' : img, 'pred' : pred.item(), 'gt' : label}
    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

count = sum(x == y for x, y in zip(preds, labels))
print("COUNT", count)
# print("ACCURACY = " ,count/)

results.to_csv('/home/ubuntu/local/Resnet50/results/weighted/noun/N_split'+str(split)+'.csv')
