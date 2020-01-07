import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from unet.unet import Unet
from os import listdir
from os import getcwd
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from data import ProcessData




VALIDATION_SET_PERCENTAGE = 0.1
BATCH_SIZE = 1
LEARNING_RATE = 0.01
EPOCH = 10
DATA_POINTS = 2000


device = 'cuda' if torch.cuda.is_available() else 'cpu'


######## Setting up the data ########
# current_path = getcwd()
# img_path = current_path + '/data/train/'
# mask_path = current_path + '/data/train_masks/'

img_path = '/floyd/input/data/train/'
mask_path = '/floyd/input/data/train_masks/'

print('Formatting data....')
img_data, mask_data = ProcessData(img_path, mask_path, DATA_POINTS).getData()


print('Preprocessing finish....')


######## Setting up the model ########
model = Unet(3,1)
loss = nn.BCEWithLogitsLoss()
optimize = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(device)

if device == 'cuda':
    model = torch.nn.DataParallel(model)

print('Model parameters set....')

for epoch in range(EPOCH):

    for i in range(DATA_POINTS):
        independent_var = img_data[i].unsqueeze(0)
        dependent_var = mask_data[i].unsqueeze(0)

        independent_var = independent_var.to(device=device, dtype=torch.float32)
        dependent_var = dependent_var.to(device=device, dtype=torch.float32)

        model.zero_grad()
        optimize.zero_grad()

        pred = model(independent_var)
        l = loss(pred, dependent_var)
        l.backward()
        optimize.step()
        print('EPOCH :', epoch, 'Loss : ', l.item())



torch.save(model.state_dict(), './unet_state.mdl')
print('Model state saved....')
torch.save(model, './unet_model.mdl')
print('Model saved....')
