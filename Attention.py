import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads=2, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), "dimentionality Must be divisbale by the number of heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.heads_dim = d_out // num_heads
        self.weight_q = torch.nn.Linear(d_in , d_out, bias=qkv_bias)
        self.weight_k = torch.nn.Linear(d_in , d_out, bias=qkv_bias)
        self.weight_v = torch.nn.Linear(d_in , d_out, bias=qkv_bias)
        self.out_projection = torch.nn.Linear(d_in , d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        # mask goes here

    def forward(self, batch):
        b, number_tokens, d_z = batch.shape

        queries = self.weight_q(batch)
        keys =  self.weight_k(batch)
        values = self.weight_v(batch)
        ## Adding a dimention equal to number of heads and use view to reshape it(format a)
        queries = queries.view(b, number_tokens, self.num_heads, self.heads_dim)
        keys = keys.view(b, number_tokens, self.num_heads, self.heads_dim)
        values = values.view(b, number_tokens, self.num_heads, self.heads_dim)

        # transpose to make MM work after the view dimentions were altered
        #(format b)
        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        ## caluculating attn_scores (keys go format c)
        attention_score = queries @ keys.transpose(2,3)
        #mask
        ## after this annt_wieghts is in format c
        attention_weights = torch.softmax(attention_score / keys.shape[-1], dim= 1)
        attention_weights = self.dropout(attention_weights)
        ##(annt_wieghts put matrix in format d) then transpose puts bac k in format a)
        context_vector = (attention_weights @ values).transpose(1,2)
        ##match desired output dimentions using reshape (keep values same jsut change shape)
        context_vector = context_vector.reshape(b, number_tokens, self.d_out)
        context_vector - self.out_projection(context_vector)

        return context_vector
'''
Testing 
'''
inputs = torch.tensor([[.5 ,.2 , .3 , .4],
           [.5 , .6 , .7 , .8],
           [.4 , .6 , .2 ,.8 ],
           [ .9 , .7 , .5 , .3],
           [.10, .11, .12, .13]])
batch = torch.stack((inputs, inputs), dim=0)
batch_size, context_legnth, d_in = batch.shape
print(d_in)
print(batch_size)
print(context_legnth)
d_out = 4
dropout = 0.2
context_length = 4
num_heads = 2
qkv_bias = False
mha = Attention(d_in, d_out, context_legnth, num_heads , dropout, qkv_bias)
c_v = mha(batch)
print(c_v) 