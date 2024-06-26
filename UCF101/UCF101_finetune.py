import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os

from UCF101_model import VideoRecog_Model
from UCF101_training_helper import train, test
from UCF101_dataloaders import get_dataloaders

import argparse

parser = argparse.ArgumentParser()

# MODEL PARAMS
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--fix_base', type=bool, default=False)

#DATASET PARAMS
parser.add_argument('--val_split', type=float, default=0.05)
parser.add_argument('--num_frames', type=int, default=16)
parser.add_argument('--clip_steps', type=int, default=50)
parser.add_argument('--crop_size', type=int, default=112)
parser.add_argument('--video_dir', type=str, default='hmdb51_video_data/')
parser.add_argument('--split_dir', type=str, default='hmdb51_test_train_splits')

#TRAINING PARAMS
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-2)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--lr_red_step', type=int, default=3)
parser.add_argument('--total_epochs', type=int, default=10)
parser.add_argument('--ckpt_freq', type=int, default=5)
parser.add_argument('--patience', type=int, default=2)
parser.add_argument('--lr_factor', type=float, default=0.5)
parser.add_argument('--min_lr', type=float, default=1e-8)

hmdb_args = parser.parse_args()

lr = hmdb_args.lr
gamma = hmdb_args.gamma
lr_red_step = hmdb_args.lr_red_step
total_epochs = hmdb_args.total_epochs
config = {}
num_workers = hmdb_args.num_workers
patience = hmdb_args.patience
lr_factor = hmdb_args.lr_factor
min_lr = hmdb_args.min_lr

train_loader, val_loader, test_loader = get_dataloaders(hmdb_args)

model = VideoRecog_Model(hmdb_args)

if hmdb_args.fix_base:
    for param in model.base_model.parameters():
        param.requires_grad = False

#model = torch.load('hmdb51_finetune.pth')
print(model)

if torch.cuda.is_available():
   model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = StepLR(optimizer, step_size=lr_red_step, gamma=gamma)
#scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=patience, factor=lr_factor, min_lr=min_lr)

print("Launching Action Recognition Model training")
os.makedirs('ckpts', exist_ok=True)
for epoch in range(1, total_epochs + 1):
    train(config, model, train_loader, optimizer, epoch)
    val_loss = test(model, val_loader, text="Validation")
    if (epoch+1) % hmdb_args.ckpt_freq == 0:
        ckpt_path = os.path.join('ckpts', f'HMDB51-{epoch+1}epochs.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f'Checkpoint saved at {ckpt_path}')
    scheduler.step()

test(model, test_loader, text="Test")

#torch.save(model.state_dict(), 'hmdb51_finetune_state_dict.pth')
#torch.save(model, 'hmdb51_finetune.pth')