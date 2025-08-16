import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.skip  = nn.Identity() if (in_c == out_c and stride==1) else nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c),
        )
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        s = self.skip(x)
        return F.relu(y + s, inplace=True)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.b1 = BasicBlock(32, 64, stride=2)
        self.b2 = BasicBlock(64, 128, stride=2)
        self.b3 = BasicBlock(128, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 128
    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        x = self.pool(x).flatten(1)
        return x

class TTTModel(nn.Module):
    def __init__(self, num_classes=10, rot_classes=4):
        super().__init__()
        self.encoder = Encoder()
        self.main_head = nn.Linear(self.encoder.out_dim, num_classes)
        self.aux_head  = nn.Linear(self.encoder.out_dim, rot_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.main_head(h), self.aux_head(h)
