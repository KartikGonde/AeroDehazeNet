import torch
import torch.nn as nn
from vkr import VKR

class DIP(nn.Module):
    def __init__(self, channels):
        super(DIP, self).__init__()
        self.vkr1 = VKR(channels)
        self.vkr2 = VKR(channels)
        self.vkr3 = VKR(channels)

    def forward(self, x):
        vkr1 = self.vkr1(x)
        vkr2 = self.vkr2(vkr1)

        sub1 = torch.sub(vkr1, vkr2)

        vkr3 = self.vkr3(sub1)

        sub2 = torch.sub(vkr1, vkr3)    

        return sub2


if __name__ == "__main__":
    x = torch.rand(1, 3, 64, 64)
    net = DIP(3)
    out = net(x)
    