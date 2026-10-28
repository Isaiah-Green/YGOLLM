import tiktoken
import torch
class Tokenizier:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base") # tokenizer
        self.special_end_token  = "<|endoftext|>"
    def decodes(self, text):
        if hasattr(text, "tolist"):
            text = text.tolist()
        elif isinstance(text, torch.Tensor): 
            text = text.cpu().numpy().tolist()
        token_ids = [item for item in text if isinstance(item, int)]
        return self.tokenizer.decode(token_ids)
    def encode(self, text):   
        if isinstance(text, torch.Tensor):
            text = text.cpu().numpy().tolist()  
        if isinstance(text, list):
            text = " ".join(text)
        return self.tokenizer.encode(text , allowed_special={"<|endoftext|>"})
    def end_token(self):
        end_token = self.encode(self.special_end_token)[0]
        return end_token
    @property
    def n_vocab(self):
        return self.tokenizer.n_vocab
'''
Testing 
import sys
import io
sys.stdout  = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
token = Tokenizier()
sample_text = ""
with open("pre_training.json", mode="r") as f:
    for text in f:
        sample_text += text
print(token.encode(sample_text))      
print(token.decodes(token.encode(sample_text))) 
'''