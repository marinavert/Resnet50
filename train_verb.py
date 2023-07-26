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
import random 
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default=1)
parser.add_argument("--batch_size", default=64)
parser.add_argument("--gpu", default=2)
parser.add_argument("--use_weights", action="store_true", default=True)
parser.add_argument("--name", type=str) #verb_split_

args = parser.parse_args()

batch_size = int(args.batch_size)
gpu = int(args.gpu)
split = args.split
import dataset

np.random.seed(13)
print("TRAINING FOR SPLIT ", split)
train_data = dataset.GTEADataset(mode="train", csv_dir="./csv/verb/split"+str(split), transform=dataset.img_transforms)
val_data = dataset.GTEADataset(mode="val", csv_dir="./csv/verb/split"+str(split), transform=dataset.img_transforms_val)


train_loader = DataLoader(train_data, batch_size,
                            shuffle=True, num_workers=4) 

val_loader = DataLoader(val_data, batch_size,
                            shuffle=True, num_workers=4) 

dataloaders = {"train": train_loader, "val": val_loader}
dataset_sizes = {"train": len(train_data), "val": len(val_data)}
device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
print("DEVICE = ", device)


def train_model(model, criterion, optimizer, scheduler, batch_size, num_epochs=100):
    wandb.init(project='resnet50_classification', name="w_"+args.name+str(split))
    since = time.time()
    patience = 3
    counter = 0

    wandb.config.learning_rate = 0.001
    wandb.config.batch_size = batch_size
    wandb.config.num_epochs = num_epochs
    wandb.config.patience = patience

    # Create a temporary directory to save training checkpoints

    torch.save(model.state_dict(), '/home/ubuntu/local/Resnet50/models/weighted/verb_split'+str(split)+'.pt')
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, name in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    labels = torch.squeeze(labels, dim=1)
                    criterion.to(device)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == "train":
                wandb.log({"Train Loss": epoch_loss, "Train Accuracy": epoch_acc})
            if phase == "val":
                wandb.log({"Validation Loss": epoch_loss, "Validation Accuracy": epoch_acc})
            


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                counter = 0
                torch.save(model.state_dict(), '/home/ubuntu/local/Resnet50/models/weighted/verb_split'+str(split)+'.pt')
            if phase == 'val' and epoch_acc < best_acc:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered. No improvement for {} epochs.".format(patience))
                    break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load('/home/ubuntu/local/Resnet50/models/weighted/verb_split'+str(split)+'.pt'))
    return model

model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, 11)

model_ft = model_ft.to(device)

class_weights = [0.747403,
      0.848367,
      0.584442,
      1.386730,
      2.256468,
      1.725615,
      2.197087,
      1.148780,
     10.671565,
      1.395593,
     0.345711]

class_weights=torch.FloatTensor(class_weights)

criterion = nn.CrossEntropyLoss()
if args.use_weights:
    print("USE WEIGHTS")
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, batch_size=batch_size,
                       num_epochs=100)

