import torch
import torch.nn as nn
import torch.nn.functional as F


class SKFF(nn.Module):
    def __init__(self, in_channels, height=2,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.Hardswish())    #nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        # print(n_feats)
        
    
        inp_feats = torch.cat(inp_feats, dim=1)
        # print(inp_feats.shape)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        # print(inp_feats)
        feats_U = torch.sum(inp_feats, dim=1)
        # print(feats_U)
        feats_S = self.avg_pool(feats_U)
        # print(feats_S.shape)
        feats_Z = self.conv_du(feats_S)
        # print(feats_Z.shape)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = inp_feats*attention_vectors
        # print(feats_V.shape)
        feats_V1, feats_V2 = torch.chunk(feats_V, 2, dim=1)
        feats_V1 = torch.squeeze(feats_V1, dim=1)
        feats_V2 = torch.squeeze(feats_V2, dim=1)
        # print(feats_V1.shape, feats_V2.shape)

        # feats_V = torch.matmul(feats_V1, feats_V2)
        # print(feats_V.shape)
        
        return feats_V1, feats_V2        

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.skff = SKFF(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        # print(q.shape, k.shape)
        q, k = self.skff([q, k])

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        # print(q.shape, k.shape, v.shape)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out

if __name__ == "__main__":
    # x = [torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)]
    x = torch.rand(1, 3, 64, 64)
    # net = SKFF(3)
    net = MDTA(3, 1)
    out = net(x)