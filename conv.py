import torch
from test import test

def conv(a, b):
    # this function does convolution over 2 discrete lists (distributions)
    # should be identical to `numpy.convolve` except taking lists as params instead of numpy.ndarray
    len_a = len(a)
    len_b = len(b)
    min_len = min(len_a, len_b)

    def local_conv(pos):
        nonlocal a, b, min_len
        # if pos >= min_len:
        a_slice = slice(
            max(pos-len_b+1, 0), min(max(pos-len_b+1, 0)+min(pos+1, min_len), len_a), 1)
        b_slice = slice(
            max(pos-len_a+1, 0), min(max(pos-len_a+1, 0)+min(pos+1, min_len), len_b), 1)
        return sum(x*y for x, y in zip(reversed(a.__getitem__(a_slice)), b.__getitem__(b_slice)))
    
    return [local_conv(pos) for pos in range(len_a+len_b-1)]

def _conv(a,b):
    len_a = len(a)
    len_b = len(b)
    min_len = min(len_a,len_b)
    return [
        sum(
            x*y for x, y in zip(
                reversed(a[
                    max(pos-len_b+1, 0): 
                    min(max(pos-len_b+1, 0)+min(pos+1, min_len), len_a): 
                    1]), 
                b[
                    max(pos-len_a+1, 0): 
                    min(max(pos-len_a+1, 0)+min(pos+1, min_len), len_b): 
                    1])) 
        for pos in range(len_a+len_b-1)]

def conv1d(input,weight,bias,stride,padding,dilation):
    # input: (bs), in_channel, input_size
    # weight: out_channel, in_channel, weight_size
    # bias: out_channel
    # output: (bs), out_channel, output_size
    if len(input.shape) == 2:
        input = torch.unsqueeze(input,dim=0)
        batch_squeeze_flag = True
    else:
        batch_squeeze_flag = False

    batch_size, in_channel, input_size = input.shape
    out_channel, _, weight_size = weight.shape
    assert _ == in_channel

    dilated_weight_size = weight_size*dilation-(dilation-1)
    output_size = (input_size + 2*padding - dilated_weight_size) / stride + 1

    input_pad = torch.nn.functional.pad(input,(padding,padding),mode='constant',value=0.0) # pad from the last dimension

    answer = torch.zeros([batch_size,out_channel,output_size])
    for i in range(output_size):
        input_slice = input_pad[:,:,i*stride:i*stride+dilated_weight_size:dilation] # (bs), in_channel, weight_size
        answer[:,:,i] = torch.einsum('abc,dbc->ad',[input_slice,weight]) + bias
    
    if batch_squeeze_flag:
        answer = torch.squeeze(answer,dim=0)
    return answer


datas = [{'input':torch.randn([4,20]),'weight':torch.randn([8,4,3]),'bias':torch.randn([8]),'stride':1,'padding':0,'dilation':1},
        {'input':torch.randn([5,4,20]),'weight':torch.randn([8,4,3]),'bias':torch.randn([8]),'stride':1,'padding':0,'dilation':1},
        {'input':torch.randn([5,4,20]),'weight':torch.randn([8,4,3]),'bias':torch.randn([8]),'stride':2,'padding':0,'dilation':1},
        {'input':torch.randn([5,4,20]),'weight':torch.randn([8,4,3]),'bias':torch.randn([8]),'stride':2,'padding':3,'dilation':1},
        {'input':torch.randn([5,4,20]),'weight':torch.randn([8,4,3]),'bias':torch.randn([8]),'stride':1,'padding':3,'dilation':2},
        {'input':torch.randn([5,4,20]),'weight':torch.randn([8,4,3]),'bias':torch.randn([8]),'stride':2,'padding':3,'dilation':2},
]
tester = test(torch.nn.functional.conv1d, conv1d)
tester.print_batch_test(datas,input_is_param_dict=True)


