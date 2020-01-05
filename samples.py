import torch
from unet.unet import UnetLite
import matplotlib.pyplot as plt
import numpy as np
import os
from data import ProcessData

device= 'gpu' if torch.cuda.is_available() else 'cpu'
path = './unet_model.mdl'
current_path = os.getcwd()
img_path = current_path + '/data/train/'
mask_path = current_path + '/data/train_masks/'

model = torch.load(path, map_location=device)

if isinstance(model, torch.nn.DataParallel):
    model = model.module
else:
    model = model


img_data, mask_data = ProcessData(img_path, mask_path, 1).getData()
segmented = model(img_data.to(device=device, dtype=torch.float32)).detach().cpu().numpy()
segmented = (segmented).astype(np.uint8)
print(segmented[0].shape)
plt.subplot(1,1,1)
plt.axis("off")
plt.title("Compare")

plt.imshow(segmented[0].squeeze())
plt.show()
