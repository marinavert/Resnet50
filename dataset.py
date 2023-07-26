import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import Compose, ToTensor, Resize, CenterCrop, RandomHorizontalFlip, RandomRotation, Normalize
from PIL import Image
import glob

class GTEADataset(Dataset):
    def __init__(
        self,
        mode: str = "train",
        csv_dir: str = "./csv",
        img_dir: str = "./GTEA",
        transform=ToTensor()):
        """
        Args :
            mode (str) : train, val or test (test is the whole dataset for feature extraction)
            csv_dir (str) : directory to the folder where the csv files with the annotations are
            img_dir (str) : directory to the folder where the train/val folders are stored
            transform (callable, optional) : Optional transform to be applied
                 on a sample
        """
        super().__init__()

        self.mode = mode
        if mode == "train":
            self.annotation_csv = pd.read_csv(os.path.join(csv_dir, "train.csv"))
        else:
            self.annotation_csv = pd.read_csv(os.path.join(csv_dir, "val.csv"))
        # else : 
        #     print("Whole dataset to extract features")
        #     self.train = pd.read_csv(os.path.join(csv_dir, "train.csv"))
        #     self.train["frame_path"] = "./csv/train/" + self.train["frame_path"].astype(str)
        #     self.val = pd.read_csv(os.path.join(csv_dir, "val.csv"))
        #     self.train["frame_path"] = "./csv/val/" + self.train["frame_path"].astype(str)
        #     self.test = [self.train, self.val]
        #     self.annotation_csv = pd.concat(self.test)

        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.annotation_csv)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_dir, "train", self.annotation_csv.iloc[idx, 0])
        # load each image in PIL format for compatibility with transforms
        image = Image.open(img_name).convert("RGB")
        label = np.array(self.annotation_csv.iloc[idx, 1])
        label = torch.tensor(label.astype('int')).view(1).to(torch.int64)
        name = self.annotation_csv.iloc[idx, 0]
        # Apply the transforms
        image = self.transform(image)

        sample = [image, label, name]
        return sample

# Test out the transforms on an image (images need to be made the same size for the dataset to work)
# Apply some image augmentation on the training set (rotation, flip)
# Normalize using imagenet RGB mean and std

img_transforms = Compose([Resize(224),
                            CenterCrop(224),
                            RandomHorizontalFlip(),
                            RandomRotation(20),
                            ToTensor(),
                            Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

img_transforms_val = Compose([Resize(224),
                                CenterCrop(224),
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])



import os
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

class VideoFramesDataset(Dataset):
    def __init__(self, video_path, video, png_path , transform=None):
        """
        Args 
            video_path : path to the hand frames "/home/ubuntu/local/Resnet50/GTEA/train" (./val)
            video : which video (S1_Cheese_C1 ...)
            png_path : path to the original frames to get the list of frame names "/home/ubuntu/data/HAR_datasets/GTEA/png"

        """
        self.video_path = video_path 
        self.video = video
        self.png_path = png_path
        self.frames = sorted(os.listdir(os.path.join(png_path,video)))
        self.transform = transform

    
    def __len__(self):
        return 1  # Return 1 to indicate the number of videos
    
    def __getitem__(self, idx):
        vid_frames = []
        for frame_name in self.frames:
            frame_path = self.video_path + "/" + self.video + "_" + frame_name
            # image = read_image(frame_path)  # Use torchvision's read_image to load the frame
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            
            vid_frames.append(image)
        
        video_frames = torch.stack(vid_frames)
        
        return video_frames

class VideoFramesDatasetNew(Dataset):
    def __init__(self, video_path, video, png_path , transform=None):
        """
        Args 
            video_path : path to the hand frames "/home/ubuntu/local/Resnet50/GTEA/train" (./val)
            video : which video (S1_Cheese_C1 ...)
            png_path : path to the original frames to get the list of frame names "/home/ubuntu/data/HAR_datasets/GTEA/png"

        """
        self.video_path = video_path 
        self.video = video
        self.png_path = png_path
        self.frames = sorted(os.listdir(os.path.join(png_path,video)))
        self.transform = transform

    
    def __len__(self):
        return len(self.frames)  # Return 1 to indicate the number of videos
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frame_path = self.video_path + "/" + self.video + "_" + self.frames[idx]
        image = Image.open(frame_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return [image, frame_path]

# # Example usage
# video_path = '/path/to/your/video/folder'
# batch_size = 1
# num_workers = 4

# # Define transforms (if needed)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # Add more transforms as needed
# ])

# Create the dataset
# dataset = VideoFramesDataset(video_path, transform=transform)

# # Create the dataloader
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)




# val_data = GTEADataset(mode="val", transform=img_transforms_val)

# train_loader = DataLoader(val_data, batch_size=1,
#                             shuffle=False, num_workers=1) 

# for index, (images, labels, name) in enumerate(train_loader):
#     print(type(name))

"""
dataset_dir = "/home/ubuntu/data/HAR_datasets/GTEA/png"
csv_dir = "/home/ubuntu/data/video_feature_extractor/csv/gtea.csv"
spatial_transform = img_transforms_val
"""