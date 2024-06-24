from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision.models.video import r3d_18
import time
import av
import transforms as T
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os
import numpy as np

from HMDB51_model import VideoRecog_Model

model = torch.load('hmdb51_finetune.pth')

classes = ['brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 'climb', 'climb_stairs',
            'dive', 'draw_sword', 'dribble', 'drink', 'eat', 'fall_floor', 'fencing',
            'flic_flac', 'golf', 'handstand', 'hit', 'hug', 'jump', 'kick', 'kick_ball',
            'kiss', 'laugh', 'pick', 'pour', 'pullup', 'punch', 'push', 'pushup',
            'ride_bike', 'ride_horse', 'run', 'shake_hands', 'shoot_ball', 'shoot_bow',
            'shoot_gun', 'sit', 'situp', 'smile', 'smoke', 'somersault', 'stand',
            'swing_baseball', 'sword', 'sword_exercise', 'talk', 'throw', 'turn', 'walk', 'wave']

test_tfms =  torchvision.transforms.Compose([
                                             T.ToFloatTensorInZeroOne(),
                                             T.Resize((128, 171)),
                                             T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                             T.CenterCrop((112, 112))
                                             ])

num_frames = 16
clip_steps = 50
num_workers = 8

hmdb51_test = torchvision.datasets.HMDB51('hmdb51_video_data/', 'hmdb51_test_train_splits/', num_frames,
                                                step_between_clips = clip_steps, fold=1, train=False,
                                                transform=test_tfms, num_workers=num_workers)

bs = 4
kwargs = {'num_workers':num_workers, 'pin_memory':True} if torch.cuda.is_available() else {'num_workers':num_workers}
test_loader  = DataLoader(hmdb51_test, batch_size=bs, shuffle=False, **kwargs)

y_pred = []
y_true = []
num_finished = 0

# iterate over test dataset
for inputs, labels in test_loader:
    inputs = inputs.to(torch.device('cuda'))
    labels = labels.to(torch.device('cuda'))
    output = model(inputs)
    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output) # Save Prediction
    labels = labels.data.cpu().numpy()
    y_true.extend(labels) # Save GT

cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                    columns = [i for i in classes])
plt.figure(figsize = (24, 14))
sn.heatmap(df_cm, annot=True)
plt.title('Confusion Matrix for HMDB51')

os.makedirs('figs', exist_ok=True)
plt.savefig(f'figs/HMDB51_CM.png')