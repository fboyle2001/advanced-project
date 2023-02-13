
from models.scr import scr_resnet
import torch

model = scr_resnet.SupConResNet()
batch = torch.zeros(7, 3, 32, 32)

o = model.features(batch)

print(o.shape)