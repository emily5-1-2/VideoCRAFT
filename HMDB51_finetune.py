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

from HMDB51_model import VideoRecog_Model

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

hmdb_args = parser.parse_args()
  
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def update_acc(self, val, n=1):
        self.val = val/n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train(config, model, loader, optimizer, epoch):
    model.train()
    config = {}
    config['log_interval'] = 100
    correct = 0
    total_loss = 0.0
    flag = 0
    Loss, Acc = AverageMeter(), AverageMeter()
    start = time.time()
    for batch_id, data in enumerate(loader):
        data, target = data[0], data[-1]
        # print("here")

        if torch.cuda.is_available():
           data = data.cuda()
           target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        Loss.update(loss.item(), data.size(0))

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        num_corrects = pred.eq(target.view_as(pred)).sum().item()
        correct += num_corrects

        Acc.update_acc(num_corrects, data.size(0))

        if flag!= 0 and batch_id%config['log_interval'] == 0:
           print('Train Epoch: {} Batch [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.0f})%'.format(
                epoch, batch_id * len(data), len(loader.dataset),
                100. * batch_id / len(loader), Loss.avg, correct, Acc.count, 100. * Acc.avg))
        flag = 1

    #total_loss /= len(loader.dataset) 
    print('Train Epoch: {} Average Loss: {:.6f} Average Accuracy: {}/{} ({:.0f})%'.format(
         epoch, Loss.avg, correct, Acc.count, 100. * Acc.avg ))
    print(f"Takes {time.time() - start}")

def test(config, model, loader, text='Validation'):
    model.eval()
    correct = 0
    total_loss = 0.0
    Loss, Acc = AverageMeter(), AverageMeter()
    with torch.no_grad():
         for batch_id, data in enumerate(loader):
             data, target = data[0], data[-1]

             if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

             output = model(data)
             loss = F.cross_entropy(output, target)
             total_loss += loss.item()

             Loss.update(loss.item(), data.size(0))

             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
             num_corrects = pred.eq(target.view_as(pred)).sum().item()
             correct += num_corrects

             Acc.update_acc(num_corrects, data.size(0))
           
    total_loss /= len(loader.dataset)
    print(text + ' Average Loss: {:.6f} Average Accuracy: {}/{} ({:.0f})%'.format(
         Loss.avg, correct, Acc.count , 100. * Acc.avg ))
    
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
#kwargs = {'num_workers':num_workers}
#kwargs = {}

hmdb51_train_v1, hmdb51_val_v1 = random_split(hmdb51_train, [total_train_samples - total_val_samples,
                                                                       total_val_samples])

#hmdb51_train_v1.video_clips.compute_clips(16, 1, frame_rate=30)
#hmdb51_val_v1.video_clips.compute_clips(16, 1, frame_rate=30)
#hmdb51_test.video_clips.compute_clips(16, 1, frame_rate=30)

#train_sampler = RandomClipSampler(hmdb51_train_v1.video_clips, 5)
#test_sampler = UniformClipSampler(hmdb51_test.video_clips, 5)
  
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

scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

print("Launching Action Recognition Model training")
for epoch in range(1, total_epochs + 1):
    train(config, model, train_loader, optimizer, epoch)
    test(config, model, val_loader, text="Validation")
    scheduler.step()

test(config, model, test_loader, text="Test")

torch.save(model.state_dict(), 'hmdb51_finetune_state_dict.pth')
torch.save(model, 'hmdb51_finetune.pth')
  