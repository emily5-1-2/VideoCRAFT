import torch
from torch.nn import functional as F
import time

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

def test(model, loader, text='Validation'):
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
    
    return Loss.avg