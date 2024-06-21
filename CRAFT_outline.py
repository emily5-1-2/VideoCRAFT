import torch
import torch.nn as nn
import torchvision
from torchvision.models.video import r3d_18
import os
from tqdm.auto import tqdm
import numpy as np
import cv2
import av
import random
import transforms as T
from craft_torch import *

tfms =  torchvision.transforms.Compose([
                                        T.ToFloatTensorInZeroOne(),
                                        T.Resize((128, 171)),
                                        T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                        T.CenterCrop((112, 112))
                                        ])


vids = []
class_name = "clap"
folder_path = "video_data/" + class_name
for filename in os.listdir(folder_path):
    f = os.path.join(folder_path, filename)
    frames = []
    cap = cv2.VideoCapture(f)
    ret = True
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret: frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    video = np.stack(frames, axis=0)
    video = torch.from_numpy(video)
    vids.append(video)

vids = [tfms(v) for v in vids]

vid_clips_8sec = []
for v in vids:
    count = 0
    for clip in torch.split(v, 8, dim=1):
        if clip.shape[1] == 8:
            count += 1
            vid_clips_8sec.append(clip)
        if count == 2: break

vid_clips_8sec = torch.from_numpy(np.array(vid_clips_8sec))

mean=[0.43216, 0.394666, 0.37645]
std=[0.22803, 0.22145, 0.216989]

import imageio
import numpy as np
from IPython.display import Image

# mean and std currently manually defined
def unnormalize_img(img):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)

# Video_tensor dims (num_frames, height, width, num_channels)
def create_gif(video_tensor, filename="sample.gif"):
    """Prepares a GIF from a video tensor.

    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame)#.numpy())
        frames.append(frame_unnormalized)
    kargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename

device = 'cuda'

model = r3d_18(weights='KINETICS400_V1')
model = model.eval().to(device)

# get first part of model
g = nn.Sequential(*(list(model.children())[:-2]))

    
craft = Craft(input_to_latent = g,
              number_of_concepts = 10,
              patch_size = 56,
              batch_size = 4,
              device = device)

# now we can start fit the concept using our rabbit images
# CRAFT will (1) create the patches, (2) find the concept
# and (3) return the crops (crops), the embedding of the crops (crops_u), and the concept bank (w)
crops, crops_u, w = craft.fit(vid_clips_8sec[:100])
crops = np.moveaxis(torch_to_numpy(crops), 1, -1)

'''
f_importances = craft.estimate_importance(f_images_preprocessed[:100], class_id=flamingo_class)
f_images_u = craft.transform(f_images_preprocessed[:100])

f_images_u.shape

import matplotlib.pyplot as plt

plt.bar(range(len(f_importances)), f_importances)
plt.xticks(range(len(f_importances)))
plt.title("Concept Importance")

most_important_concepts = np.argsort(f_importances)[::-1][:5]

for c_id in most_important_concepts:
  print("Concept", c_id, " has an importance value of ", f_importances[c_id])'''

from math import ceil
nb_crops = 10

for c_id in range(10):
  best_crops_ids = np.argsort(crops_u[:, c_id])[::-1][:nb_crops]
  print(best_crops_ids)
  best_crops = crops[best_crops_ids]

  for i in range(nb_crops):
    create_gif(best_crops[i], "concepts/Concept_{}_crop_{}.gif".format(c_id, i))
