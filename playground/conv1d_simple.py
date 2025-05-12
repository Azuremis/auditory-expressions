import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


#
#
#
cov = torch.nn.Conv1d(
    in_channels=1,
    out_channels=1,
    kernel_size=1,
    padding=0,
    stride=1,
    bias=False
)

#
#
#
with torch.no_grad():
    cov.weight[:] = 0.5


#
#
#
bar = torch.tensor([[1.4, 2.2, 0.8, 0.6, 1.2, 0.4, 5.0]])
out = cov(bar)
print(out)




