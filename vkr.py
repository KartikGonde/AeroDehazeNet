import torch
import torch.nn as nn
import torch.nn.functional as F

class DPW(nn.Module):
    def __init__(self, channels, k):
        super(DPW, self).__init__()

        self.dpw = nn.Conv2d(channels, channels*k, kernel_size=k, bias=False, groups=channels)

    def forward(self, x):
         out = self.dpw(x)
         return out
    
class VKR(nn.Module):
    def __init__(self, channels):
        super(VKR, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(128, channels, kernel_size=1, bias=False)

        self.conv_4dp5 = nn.Conv2d(channels, channels, kernel_size=5, bias=False, groups=channels, padding=2)
        self.conv_4dp3 = nn.Conv2d(channels, channels, kernel_size=3, bias=False, groups=channels, padding=1)
        self.conv_4dp1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False, groups=channels)
        
        self.conv5 = nn.Conv2d(channels*3, channels, kernel_size=1, bias=False)

    def forward(self, x):
        first = self.conv1(x)
        first = nn.Hardswish()(first)                              #nn.LeakyReLU(0.1)(first)

        second = self.conv2(first)
        second = nn.Hardswish()(second)                                                #nn.LeakyReLU(0.1)(second)

        Vm = self.conv3(second)
        Vm = nn.Hardswish()(Vm)                                                   #nn.LeakyReLU(0.1)(Vm)
        conv_41 = self.conv_4dp5(Vm)
        conv_42 = self.conv_4dp3(Vm)
        conv_43 = self.conv_4dp1(Vm)

        concat = torch.cat((conv_41, conv_42, conv_43), dim=1)
        concat = self.conv5(concat)
        concat = nn.Hardswish()(concat)                                                               #nn.LeakyReLU(0.1)(concat)

        Vm_hat = Vm + concat
        return Vm_hat

if __name__ == "__main__": 
    x = torch.rand(1, 3, 64, 64)
    net = VKR(3)
    out = net(x)
    