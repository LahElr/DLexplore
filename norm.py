import torch
import torch.nn as nn
import torch.nn.functional as F
from test import test

x = torch.rand([4,5,10])

class layer_norm(nn.Module):
    def __init__(self,eps):
        super(layer_norm,self).__init__()
        self.eps = eps
    def forward(self,x):
        mu = torch.mean(x,dim=-1,keepdim=True)
        sigma = torch.mean((x-mu)**2, dim=-1,keepdim=True)
        x = (x-mu)/(torch.sqrt(sigma+self.eps))
        return x

tester = test(tester = torch.nn.LayerNorm(10,eps = 1e-5, elementwise_affine=False),testee = layer_norm(eps = 1e-5),eps=1e-5)
print(tester.test(x))

class batch_norm(nn.Module):
    def __init__(self,eps):
        super(batch_norm,self).__init__()
        self.eps = eps
    def forward(self,x):
        mu = torch.mean(x,dim=1,keepdim=True)
        sigma = torch.mean((x-mu)**2,dim=1,keepdim=True)
        x = (x-mu)/(torch.sqrt(sigma+self.eps))
        return x

tester = test(tester = torch.nn.BatchNorm1d(5,eps = 1e-5, affine=False),testee = batch_norm(eps = 1e-5),eps=1e-5)
print(tester.test(x))

x = torch.rand([2,3,4,5])
tester = test(tester = torch.nn.BatchNorm2d(3,eps=1e-5,momentum=0,affine=False), testee = batch_norm(eps=1e-5), eps=1e-5)
print(tester.test(x))

