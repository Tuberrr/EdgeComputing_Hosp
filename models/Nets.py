import torch
from torch import nn
import torch.nn.functional as F

class CNNCifar10(nn.Module):
    def __init__(self):
        super(CNNCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:  #
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        result = {'activation' : x}
        x = x.view(-1, 16 * 5 * 5)
        result['hint'] = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        result['representation'] = x
        x = self.fc3(x)
        result['output'] = x
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc3(z)

        result = {'output': z}
        if logit:
            result['logit'] = z
        return result