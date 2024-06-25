import torch
import torch.nn as nn
import os
import numpy as np
from craft_torch import *
from HMDB51.HMDB51_model import VideoRecog_Model
from HMDB51.HMDB51_confusion_matrix import get_classes
from HMDB51.HMDB51_dataloaders import get_dataloaders
import imageio

import argparse

parser = argparse.ArgumentParser()


#DATASET PARAMS
parser.add_argument('--val_split', type=float, default=0)
parser.add_argument('--num_frames', type=int, default=8)
parser.add_argument('--clip_steps', type=int, default=50)
parser.add_argument('--crop_size', type=int, default=112)
parser.add_argument('--video_dir', type=str, default='HMDB51/hmdb51_video_data/')
parser.add_argument('--split_dir', type=str, default='HMDB51/hmdb51_test_train_splits')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=4)

hmdb_args = parser.parse_args()

train_loader, _, test_loader = get_dataloaders(hmdb_args)

device = 'cuda'
model = torch.load('HMDB51/hmdb51_finetune.pth')
model = model.eval().to(device)

#print(torchsummary.summary(model, (3, 8, 112, 112)))

mean=[0.43216, 0.394666, 0.37645]
std=[0.22803, 0.22145, 0.216989]

classes = get_classes()

def unnormalize_img(img):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)

os.makedirs('HMDB51_concepts', exist_ok=True)

for i in range(6):
    vids = {}
    for j in range(i*10, min((i+1)*10, 51)):
        vids[j] = []

    with torch.no_grad():
        for _, data in enumerate(train_loader):
            inputs = data[0]
            inputs = inputs.to(torch.device('cuda'))
            output = model(inputs)
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            for j, l in zip(inputs, output):
                if i*10 < l < (i+1)*10:
                    vids[l].append(torch.unsqueeze(j, dim=0))
        for _, data in enumerate(test_loader):
            inputs = data[0]
            inputs = inputs.to(torch.device('cuda'))
            output = model(inputs)
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            for j, l in zip(inputs, output):
                if i*10 <= l < (i+1)*10:
                    vids[l].append(torch.unsqueeze(j, dim=0))
        
    for j in range(i*10, min((i+1)*10, 51)):
        print("Generating concepts for class", classes[j])
        vid_clips_8sec = torch.cat(vids[j], dim=0)
        g = nn.Sequential(*list(model.base_model.children())[:-1])
        #print(torchsummary.summary(g, (3, 8, 112, 112)))
        craft = Craft(input_to_latent = g,
                    number_of_concepts = 10,
                    patch_size = 56,
                    batch_size = 8,
                    device = device)
        crops, crops_u, w = craft.fit(vid_clips_8sec)
        crops = np.moveaxis(torch_to_numpy(crops), 1, -1)
        nb_crops = 10
        for c_id in range(10):
            best_crops_ids = np.argsort(crops_u[:, c_id])[::-1][:nb_crops]
            best_crops = crops[best_crops_ids]

            gifs = []
            for crop in best_crops:
                frames = []
                for frame in crop:
                    frames.append(unnormalize_img(frame))
                gifs.append(frames)

            gifs = np.array(gifs)

            display = []
            for k in range(8):
                display.append(np.concatenate(gifs[:, k, :, :, :], axis=1))
            display = np.array(display)

            imageio.mimsave("HMDB51_concepts/Class_{}_Concept_{}.gif".format(classes[j], c_id), display, "GIF")