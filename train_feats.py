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

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=64)
parser.add_argument("--gpu", default=1)

args = parser.parse_args()

# use_weights = args.weights
batch_size = int(args.batch_size)
gpu = int(args.gpu)

import dataset

np.random.seed(13)

train_data = dataset.GTEADataset(mode="train", csv_dir="./csv", transform=dataset.img_transforms)
val_data = dataset.GTEADataset(mode="val", csv_dir="./csv", transform=dataset.img_transforms_val)


train_loader = DataLoader(train_data, batch_size,
                            shuffle=True, num_workers=4) 

val_loader = DataLoader(val_data, batch_size,
                            shuffle=True, num_workers=4) 

dataloaders = {"train": train_loader, "val": val_loader}
dataset_sizes = {"train": len(train_data), "val": len(val_data)}
device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
print("DEVICE = ", device)


def train_model(model, criterion, optimizer, scheduler, batch_size, num_epochs=100):
    wandb.init(project='hand_resnet50', name='weighted_feats_extract_train')
    since = time.time()
    patience = 3
    counter = 0

    wandb.config.learning_rate = 0.001
    wandb.config.batch_size = batch_size
    wandb.config.num_epochs = num_epochs
    wandb.config.patience = patience

    # Create a temporary directory to save training checkpoints

    torch.save(model.state_dict(), '/home/ubuntu/local/Resnet50/checkpoints/weighted.pt')
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
                torch.save(model.state_dict(), '/home/ubuntu/local/Resnet50/checkpoints/weighted.pt')
            if phase == 'val' and epoch_acc < best_acc:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered. No improvement for {} epochs.".format(patience))
                    break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load('/home/ubuntu/local/Resnet50/checkpoints/weighted.pt'))
    return model

model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, 73)

model_ft = model_ft.to(device)

class_weights = [0.05209349969886704,
 5.859448301745168,
 1.1024219742974155,
 2.72445685367769,
 3.655895094251259,
 5.414426911739206,
 6.110567514677103,
 2.9704147640791474,
 0.8010107228977477,
 1.4450666419844502,
 2.263173153584112,
 1.608044082809764,
 1.0382032185131,
 3.2651887483007425,
 1.5842212075088788,
 1.5725725221595488,
 5.2163381222853324,
 3.4219178082191783,
 4.700436549751618,
 1.5276418786692758,
 1.1467552976605826,
 0.8676262191225097,
 0.8693896870475556,
 1.3161222339304532,
 1.5901105056780567,
 1.9894870978018477,
 0.7980218769167859,
 1.397842241919599,
 1.5386321080122205,
 1.1052706098899154,
 1.69067085386323,
 0.8623784798939461,
 2.1494458594341572,
 0.22301341294441987,
 0.6736058677596807,
 2.561315724714954,
 4.035280434220728,
 5.555061376979186,
 2.4724839654762847,
 2.777530688489593,
 4.550422617312737,
 6.290290088638195,
 4.035280434220728,
 4.806064337386486,
 2.72445685367769,
 3.0122515917422343,
 1.9010654490106544,
 10.184279191128507,
 1.812456466217785,
 0.7777085927770859,
 1.843705715635333,
 1.0693493150684932,
 0.9238439007071215,
 0.3400156804669295,
 0.9198703785535426,
 0.2726193282520059,
 9.505327245053273,
 0.34301501686238756,
 0.7877343020762381,
 2.5460697977821267,
 5.780266567937801,
 2.403032168693243,
 2.4166086216237135,
 1.6141121736882915,
 4.916548575027555,
 5.346746575342466,
 4.277397260273973,
 2.531004296020102,
 2.3896074079742866,
 0.40163354556563124,
 1.6643569106124407,
 2.9097940546081444,
 1.384270958017467]

class_weights=torch.FloatTensor(class_weights)
print("USE WEIGHTS")
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, batch_size=batch_size,
                       num_epochs=100)

