import torch
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchvision
import transforms as T
import os

from HMDB51_model import VideoRecog_Model
from HMDB51_training_helper import train, test

import argparse

parser = argparse.ArgumentParser()

# MODEL PARAMS
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--fix_base', type=bool, default=False)

#TRAINING PARAMS
parser.add_argument('--val_split', type=float, default=0.05)
parser.add_argument('--num_frames', type=int, default=16)
parser.add_argument('--clip_steps', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-2)
parser.add_argument('--gamma', type=float, default=0.7)
parser.add_argument('--total_epochs', type=int, default=10)
parser.add_argument('--ckpt_freq', type=int, default=5)

hmdb_args = parser.parse_args()
    
train_tfms = torchvision.transforms.Compose([
                                 T.ToFloatTensorInZeroOne(),
                                 T.Resize((128, 171)),
                                 T.RandomHorizontalFlip(),
                                 T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                 T.RandomCrop((112, 112))
                               ]) 

test_tfms =  torchvision.transforms.Compose([
                                             T.ToFloatTensorInZeroOne(),
                                             T.Resize((128, 171)),
                                             T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                             T.CenterCrop((112, 112))
                                             ])

hmdb51_train = torchvision.datasets.HMDB51('hmdb51_video_data/', 'hmdb51_test_train_splits/', hmdb_args.num_frames,
                                                step_between_clips = hmdb_args.clip_steps, fold=1, train=True,
                                                transform=train_tfms, num_workers=hmdb_args.num_workers)

hmdb51_test = torchvision.datasets.HMDB51('hmdb51_video_data/', 'hmdb51_test_train_splits/', hmdb_args.num_frames,
                                                step_between_clips = hmdb_args.clip_steps, fold=1, train=False,
                                                transform=test_tfms, num_workers=hmdb_args.num_workers)
      
total_train_samples = len(hmdb51_train)
total_val_samples = round(hmdb_args.val_split * total_train_samples)

print(f"number of train samples {total_train_samples}")
print(f"number of validation samples {total_val_samples}")
print(f"number of test samples {len(hmdb51_test)}")

bs = hmdb_args.batch_size
lr = hmdb_args.lr
gamma = hmdb_args.gamma
total_epochs = hmdb_args.total_epochs
config = {}
num_workers = hmdb_args.num_workers

kwargs = {'num_workers':num_workers, 'pin_memory':True} if torch.cuda.is_available() else {'num_workers':num_workers}

hmdb51_train_v1, hmdb51_val_v1 = random_split(hmdb51_train, [total_train_samples - total_val_samples,
                                                                       total_val_samples])
  
train_loader = DataLoader(hmdb51_train_v1, batch_size=bs, shuffle=True, **kwargs)
val_loader   = DataLoader(hmdb51_val_v1, batch_size=bs, shuffle=True, **kwargs)
test_loader  = DataLoader(hmdb51_test, batch_size=bs, shuffle=False, **kwargs)

model = VideoRecog_Model(hmdb_args)

fix_base = False

if fix_base:
    for param in model.base_model.parameters():
        param.requires_grad = False

#model = torch.load('hmdb51_finetune.pth')
print(model)

if torch.cuda.is_available():
   model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=3, factor=0.5, min_lr=1e-8)

print("Launching Action Recognition Model training")
os.makedirs('ckpts', exist_ok=True)
for epoch in range(1, total_epochs + 1):
    train(config, model, train_loader, optimizer, epoch)
    val_loss = test(model, val_loader, text="Validation")
    if (epoch+1) % hmdb_args.ckpt_freq == 0:
        ckpt_path = os.path.join('ckpts', f'ViT-{epoch+1}epochs.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f'Checkpoint saved at {ckpt_path}')
    scheduler.step(val_loss)

test(model, test_loader, text="Test")

#torch.save(model.state_dict(), 'hmdb51_finetune_state_dict.pth')
#torch.save(model, 'hmdb51_finetune.pth')