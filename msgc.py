import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GCB(nn.Module):
    def __init__(self, k, channels):
        super(GCB, self).__init__()
        self.dpw_1 = nn.Conv2d(channels, channels*k, kernel_size=k, padding=math.floor(k/2), bias=False, groups=channels)
        self.pointwise = nn.Conv2d(channels*k, channels, kernel_size=1, bias=False)

    def forward(self, x):
        level1 = self.dpw_1(x)
        level2 = self.dpw_1(x)

        level1 = nn.GELU()(level1)
        dot_prod = torch.mul(level1, level2)
        out = self.pointwise(dot_prod)

        return out

class MSGC(nn.Module):
    def __init__(self, channels):
        super(MSGC, self).__init__()

        self.ln = nn.LayerNorm(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        self.gcb_1 = GCB(1, channels)
        self.gcb_3 = GCB(3, channels)
        self.gcb_5 = GCB(5, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, h, w, c)
        first = self.ln(x)
        b1, h1, w1, c1 = first.shape
        first = first.reshape(b1, c1, h1, w1)
        pointwise = self.pointwise(first)
        level1 = self.gcb_1(pointwise)
        level2 = self.gcb_3(pointwise)
        level3 = self.gcb_5(pointwise)

        out = level1 + level2 + level3
        return out


if __name__ == "__main__":
    x = torch.rand(1, 48, 128, 128)
    net = MSGC(48)
    out = net(x)