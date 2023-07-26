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

# parser = argparse.ArgumentParser()
# parser.add_argument("--folder", type=str) # resnet50_noun2
# parser.add_argument("--pth", type=str) # name of the pt file (noun_64..)
# args = parser.parse_args()


def extract(video, model, folder):
    """
    Args :
        mode = train or val (to get correct path)
        video = S1_Cheese_C1 ...
        model = resnet
        folder = name of folder (resnet50_noun...)
    """
    ## Create the directory to save the batch features
    if not os.path.exists("/home/ubuntu/local/Resnet50/features/"+folder+"/"+video):
        os.makedirs("/home/ubuntu/local/Resnet50/features/"+folder+"/"+video)
    i = 1
    for frame_batch in dataloader:
        total_feats = []
        
        for frames in frame_batch[0].squeeze(0):
            frames = frames.to(device)
            features = model(frames.unsqueeze(0))
            features = features.view(2048).to('cpu')
            total_feats.append(features)
        intermediate_features = torch.stack(total_feats, dim=0)
    # final_features = torch.stack(total_feats, dim=0)
        torch.save(intermediate_features, "/home/ubuntu/local/Resnet50/features/"+folder+"/"+video+"/batch_"+str(i)+".pth")
        i+=1
    return total_feats

def concat_features(video, folder):
    if not os.path.exists("/home/ubuntu/local/Resnet50/features/"+folder+"/hand_feats"):
        os.makedirs("/home/ubuntu/local/Resnet50/features/"+folder+"/hand_feats")

    batch_feats = sorted(os.listdir(os.path.join("/home/ubuntu/local/Resnet50/features/"+folder, video)))
    final = torch.empty(0,2048)

    for batch in batch_feats:
        feats = torch.load("/home/ubuntu/local/Resnet50/features/"+folder+"/"+video+"/"+batch)
        final = torch.cat((final, feats), dim=0)
    torch.save(final, "/home/ubuntu/local/Resnet50/features/"+folder+"/hand_feats/"+video+".pth")


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = resnet.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
resnet.fc = nn.Linear(num_ftrs, 73)

# Load state dict 
state_dict = torch.load('/home/ubuntu/local/Resnet50/checkpoints/weighted.pt')


# Load state dict into the model
resnet.load_state_dict(state_dict)
# Remove the last fully connected layer
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)

# Set the model to evaluation mode
resnet.eval()

print("Extracting")

vids = sorted(os.listdir("/home/ubuntu/data/HAR_datasets/GTEA/png"))
for vid in vids:
    data = dataset.VideoFramesDatasetNew(video_path="/home/ubuntu/local/Resnet50/GTEA/train", video=vid, png_path="/home/ubuntu/data/HAR_datasets/GTEA/png", transform=dataset.img_transforms_val)
    dataloader = DataLoader(data, 64, shuffle=False, num_workers=4)
    print("Extracting ", vid)
    extract(video=vid, model=resnet, folder="weighted_resnet")
    concat_features(video=vid, folder="weighted_resnet")
