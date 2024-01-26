import kornia as K
import torch
import torch.nn as nn

class EdgeLoss(nn.Module):     
    def __init__(self):         
        super(EdgeLoss,self).__init__()            
         
    def forward(self,x,y):         
        edge1 = K.filters.sobel(x)         
        edge2 = K.filters.sobel(y)        
        loss = torch.mean(torch.abs(edge1 - edge2))         
        return loss 