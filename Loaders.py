from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizier
import torch
import json
class dataset(Dataset):
    def __init__(self,text, tokenizer, max_length, stride):
        self.inputs = []
        self.targets = []
        tokens = tokenizer.encode(text) 
        ######
        for i in range(0, len(tokens) - max_length, stride):
            inputs = tokens[i: max_length + i]
            targets = tokens[i+1: max_length + 1 + i]
            self.inputs.append(torch.tensor(inputs))
            self.targets.append(torch.tensor(targets))
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, x):
        return self.inputs[x], self.targets[x]

'''
Testing 
sample_text = " The architecture of modern deep learning models, particularly the Transformer, represents a massive shift in how machines "
t = Tokenizier()
context_length = 2
inputs , targets = DataSet(sample_text, t,context_length)
print(f" {inputs} \n {targets}")
'''

def Dataloader(text,batch_size=8,max_size= 256, num_workers=0, shuffle=True, drop_last=True, stride=128):
    tokenizer = Tokenizier()
    data_set = dataset(text, tokenizer,max_size, stride)
    data_loader = DataLoader(data_set , batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,num_workers=num_workers)
    return data_loader
'''Testing 

sample_text = "The architecture of modern deep learning models, particularly the Transformer, represents a massive shift in how machines This is where the dual-embedding system rescues the architecture. Instead of forcing a single matrix to learn both what a word means and where it is located, the model splits these responsibilities into two independent lookup tables. The token embedding layer functions as a vast, high-dimensional dictionary. It maps individual tokens into a dense vector space where words with similar semantic meanings sit close to one another. For instance, the vectors for  and are guided toward the same neighborhood because they share semantic traits of royalty and power. Yet, this dictionary is completely static; it doesn't care if a word is the opening hook of a novel or a closing punctuation mark."
data = Dataloader(sample_text)
data_iter = iter(data)
first = next(data_iter)
print(first)
second = next(data_iter)
print(second)
'''