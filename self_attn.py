import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.per_head_size = embedding_size // self.num_heads

        assert embedding_size == self.per_head_size *  self.num_heads

        # These are still of dimension d_model. They will be split into number of heads 
        self.W_q = nn.Linear(embedding_size, embedding_size)
        self.W_k = nn.Linear(embedding_size, embedding_size)
        self.W_v = nn.Linear(embedding_size, embedding_size)

        # Outputs of all sub-layers need to be of dimension d_model
        self.W_o = nn.Linear(embedding_size, embedding_size)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # x: batch_size num_heads sentence_length per_head_size
        x = torch.matmul(q, k.transpose(-2, -1))
        x = x / math.sqrt(self.per_head_size)
        # x: batch_size num_heads sentence_length sentence_length
    
        # Apply the mask
        # batch_size num_heads sentence_length sentence_length
        if mask is not None:
            # x[mask] = -float('inf')
            # x[torch.logical_not(mask)] = -float('inf')
            x = x.masked_fill(mask, float('-inf'))
    
        # Calculate the attention weights (softmax over the last dimension)
        x = F.softmax(x, dim=-1)
    
        # Apply the self attention to the values
        attention = torch.matmul(x, v)
    
        return attention, x


    def split_heads(self, x:torch.Tensor):
        # x:      batch_size           sentence_length embedding_size
        # return: batch_size num_heads sentence_length per_head_size
        
        return x.view(x.shape[0], -1, self.num_heads, self.per_head_size).transpose(1, 2)

    def forward(self, q, k=None, v=None, mask=None):
        # q k v: batch_size sentence_length embedding_size

        q = self.W_q(q)
        if k is not None:
            k = self.W_k(k)
        else:
            k = self.W_k(q)
        if v is not None:
            v = self.W_v(v)
        else:
            v = self.W_v(q)

        # split into multiple heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        # q k v: batch_size num_heads sentence_length per_head_size

        # self attention
        x, weight = self.scaled_dot_product_attention(q, k, v, mask)

        # concatenate heads
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.embedding_size)
        x = self.W_o(x)

        return x, weight
    

testee = MultiHeadAttention(20,4)
q = torch.randn([3,10,20])
k = torch.randn([3,10,20])
v = torch.randn([3,10,20])
mask = torch.randint(0,2,[3,4,10,10]).bool()

testee_result, testee_weight = testee(q,k,v,mask)

tester_result, tester_weight = F.multi_head_attention_forward(
    query=q.transpose(0,1).contiguous(),
    key=k.transpose(0,1).contiguous(),
    value=v.transpose(0,1).contiguous(),
    attn_mask=mask.view(-1,10,10),
    num_heads=4,
    embed_dim_to_check=20,
    in_proj_weight=torch.concat([testee.W_q.weight,testee.W_k.weight,testee.W_v.weight],dim=0),
    in_proj_bias=torch.concat([testee.W_q.bias,testee.W_k.bias,testee.W_v.bias],dim=0),
    out_proj_bias=testee.W_o.bias,
    out_proj_weight=testee.W_o.weight,
    use_separate_proj_weight=False,
    need_weights=True,
    bias_k=None,
    bias_v=None,
    add_zero_attn=False,
    dropout_p=0.0,
    average_attn_weights=False)

tester_result = tester_result.transpose(0,1)
print(tester_result.shape)
print(testee_result.shape)
print(tester_weight.shape)
print(testee_weight.shape)
print(torch.all(torch.abs(testee_result-tester_result)<1e-5))
print(torch.all(torch.abs(testee_weight-tester_weight)<1e-5))