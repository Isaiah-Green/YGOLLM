import torch.nn as nn
from Transformer import TransformerBlock
import torch
from tokenizer import Tokenizier
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.logits = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.Transformer = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range (cfg["n_layers"])]
        )
        self.dropout = nn.Dropout(cfg["dropout"])
        self.Layer_Norm = nn.LayerNorm(cfg["emb_dim"], eps=1e-5)

    def forward(self, input):
        b, size = input.shape
        embed = self.embedding(input)
        pos_embed = self.pos_embedding(torch.arange(size, device=input.device))
        x = embed + pos_embed
        ##
        x = self.dropout(x)
        x = self.Transformer(x)
        x = self.Layer_Norm(x)
        logits = self.logits(x)
        return logits

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
t = Tokenizier()
random = torch.tensor(t.encode(" The World was bright today and i like the sun"))
random2 = torch.tensor(t.encode(" The sun set was despair enducing and i "))
block  = []
block.append(random)
block.append(random2)
block = torch.stack(block, dim=0)
model = Model(cfg)
print(block.shape)
print(model(block))
'''
