import torch
from test import test

def softmax(input,dim):
    input = input - torch.max(input)
    input = torch.exp(input)
    return input / torch.sum(input,dim=dim,keepdim=True)

data = {'input':torch.rand([4,5]),'dim':0}
tester = test(torch.softmax, softmax)
tester.print_test(data,input_is_param_dict=True)


def CELoss(input,target):
    return -target*torch.log(input)

data = {'input':torch.softmax(torch.rand([4]),dim=0), 'target':torch.Tensor([0,0,1,0])}
tester = test(torch.nn.functional.cross_entropy, CELoss)
tester.print_test(data,input_is_param_dict=True)


def softmax_CELoss(input,target):
    input = input - torch.max(input)
    return -target* (input - torch.log(torch.sum(torch.exp(input))))

data = {'input':torch.rand([4]), 'target':torch.Tensor([0,0,1,0])}
tester = test(torch.nn.functional.binary_cross_entropy_with_logits, softmax_CELoss)
tester.print_test(data,input_is_param_dict=True)


def NLLLoss(input,target):
    return -torch.matmul(input,target)

data = {'input':torch.rand([4,5]), 'target':torch.Tensor([0,0,4,2]).long()}
tester = test(torch.nn.functional.nll_loss,NLLLoss)
tester.print_test(data,input_is_param_dict=True)


def SmoothL1Loss(input,target,beta=1.0):
    a = torch.abs(input-target)-0.5
    b = torch.square(input-target)*0.5

    mask = a < beta-0.5
    a[mask] = b[mask]
    return a

data = {'input':torch.rand([4]), 'target':torch.rand([4]), 'beta':1.0}
tester = test(torch.nn.functional.smooth_l1_loss,SmoothL1Loss)
tester.print_test(data,input_is_param_dict=True)