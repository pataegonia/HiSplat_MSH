from compressai.ans import BufferedRansEncoder, RansDecoder

from .pywave import DWT_2D, IDWT_2D
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch


def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x


class WLS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(WLS, self).__init__()
        self.scale_clamp = 4.0
        
        self.dwt = DWT_2D(wave='haar')      
        self.OLP = OLP(in_dim*4,out_dim)
        self.scaling_factors = nn.Parameter(torch.cat((torch.zeros(1,1,in_dim)+0.5,
                                                    torch.zeros(1,1,in_dim)+0.5,
                                                    torch.zeros(1,1,in_dim)+0.5,
                                                    torch.zeros(1,1,in_dim)),dim=2))
    def forward(self,x):
        
        x = self.dwt(x)
        b,_,h,w = x.shape
        x = x.view(b,-1,h*w).permute(0,2,1)
        scale = torch.exp(self.scaling_factors.clamp(-self.scale_clamp, self.scale_clamp))
        x = x * scale
        x = self.OLP(x)
        return x.view(b,h,w,-1).permute(0,3,1,2)
    
    
class iWLS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(iWLS, self).__init__()
        self.scale_clamp = 4.0
        self.idwt = IDWT_2D(wave='haar')      
        self.OLP = OLP(in_dim,out_dim*4)
        self.scaling_factors = nn.Parameter(torch.cat((torch.zeros(1,1,out_dim)+0.5,
                                                    torch.zeros(1,1,out_dim)+0.5,
                                                    torch.zeros(1,1,out_dim)+0.5,
                                                    torch.zeros(1,1,out_dim)),dim=2))
    def forward(self,x):
        b,_,h,w = x.shape
        x = x.view(b,-1,h*w).permute(0,2,1)
        x = self.OLP(x)
        scale = torch.exp(self.scaling_factors.clamp(-self.scale_clamp, self.scale_clamp))
        x = x / scale
        x = x.view(b,h,w,-1).permute(0,3,1,2)
        x = self.idwt(x)
        
        return x
        

class OLP(nn.Module):
    
    def __init__(self, in_features, out_dim, bias=True):
        super(OLP, self).__init__()
        self.linear = nn.Linear(in_features, out_dim, bias=bias)
        self.in_dim = in_features
        self.out_dim = out_dim
        eye_dim = min(in_features,out_dim)
        self.identity_matrix = torch.eye(eye_dim)  

    def loss(self):
        kernel_matrix = self.linear.weight
        if self.in_dim > self.out_dim:
            gram_matrix = torch.mm(kernel_matrix, kernel_matrix.t())
        else:
            gram_matrix = torch.mm(kernel_matrix.t(), kernel_matrix)
        loss_ortho = F.mse_loss(gram_matrix, self.identity_matrix.to(gram_matrix.device))
        return loss_ortho
            
    def forward(self, x):
        output = self.linear(x)
        
        return output


