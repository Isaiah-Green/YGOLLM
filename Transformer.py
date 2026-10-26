import torch
import torch.nn as nn
from Attention import Attention
class Feed_Forward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(cfg["emb_dim"] , 4 * cfg["emb_dim"]) 
                                   , torch.nn.GELU(), 
                                   nn.Linear(4 * cfg["emb_dim"] , cfg["emb_dim"]))
    def forward(self,x):
        return self.layer(x)
    
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attention = Attention(cfg["emb_dim"] ,
                                   cfg["emb_dim"],
                                   cfg["context_length"],
                                   cfg["num_heads"],
                                   cfg["dropout"],
                                   cfg["qkv_bias"])
        self.norm1 = torch.nn.LayerNorm(cfg["emb_dim"], eps=1e-5)
        self.norm2 = torch.nn.LayerNorm(cfg["emb_dim"], eps=1e-5)
        self.feed = Feed_Forward(cfg)
        self.dropout = torch.nn.Dropout(cfg["dropout"])

    def forward(self,x):
        shortcut = x
        ###norm before each step for stabiliy (pre-LN)
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = shortcut + x
        ###
        shortcut = x
        x = self.norm2(x)
        x = self.feed(x)
        x = self.dropout(x)
        x = x + shortcut

        return x

        
'''
Testing

cfg = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "num_heads": 12,
    "n_layers": 12,
    "dropout": 0.1,
    "qkv_bias": False
}
random = torch.rand(2 , 6, 768)
block  = TransformerBlock(cfg)
print(block(random))
'''
