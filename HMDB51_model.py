
import torch.nn as nn
from torchvision.models.video import r3d_18

class VideoRecog_Model(nn.Module):
  def __init__(self, args):
      super(VideoRecog_Model, self).__init__()
      self.base_model = nn.Sequential(*list(r3d_18(weights=True).children())[:-1])
      self.fc1 = nn.Linear(512, 51)
      self.dropout = nn.Dropout2d(args.dropout)

  def forward(self, x):
      out = self.base_model(x).squeeze(4).squeeze(3).squeeze(2) # output of base model is bs x 512 x 1 x 1 x 1
      out = self.fc1(out)
      return out