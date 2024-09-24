import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm, trange
from vae import VAE
from VoxelDataset import RotatedVoxelDataset
import torchvision
import torchvision.datasets as datasets
import pandas as pd
import matplotlib.pyplot as plt
from utils import train_model, validate_model, save_from_dataloader
import os

learning_rate = 0.05
momentum = 0.9
batch_size = 128
epoch_num = 100
beta = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


train_voxels = RotatedVoxelDataset(train=True)
test_voxels = RotatedVoxelDataset(train=False)
# Initialize DataLoader
train_dataloader = DataLoader(train_voxels, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_voxels, batch_size=batch_size, shuffle=True)

# Initialize Learning Rate Scheduler
# Example: Decays the learning rate by gamma every step_size epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
op_path = '/home/anurag/codes/generative_3d/original_v2/op3'

train_model(epoch_num, model, train_dataloader, device, optimizer, scheduler, beta, op_path)
validate_model(model, train_dataloader, device, op_path, 'train_validation.txt')
validate_model(model, test_dataloader, device, op_path, 'test_validation.txt')
train_dataloader = DataLoader(train_voxels, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_voxels, batch_size=1, shuffle=True)
save_from_dataloader(train_dataloader, model, device, os.path.join(op_path, 'train_reconstructions'))
save_from_dataloader(test_dataloader, model, device, os.path.join(op_path, 'test_reconstructions'))
