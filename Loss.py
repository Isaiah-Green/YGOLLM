import torch
from Model import Model
from Loaders import Dataloader
from tokenizer import Tokenizier
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device) # sends input and target to gpu or cpu for faster computing 
    logits = model(input_batch) # gets logits from model 
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_batch.flatten()) #flatten the tensors to make shapes match for calc
    return loss 

def calculate_loss(data_loader,model, device, num_batches=None):
    total_loss = 0.0
    #model.eval()
    # if there no data in the loader return null
    if(len(data_loader) == 0):
        return float("nan")
    ## if the amount of baches isnt set , make it equal the length of the dataloader
    elif num_batches == None:
        num_batches = len(data_loader)
    #otherwise choose the smaller of the 2 values 
    else:
        num_batches = min(len(data_loader) , num_batches)
    #loop through and calculate loss for each batch, then append it to the total
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
    # returns the total loss accumulated divided by the number of batches
    #model.train()
    return total_loss / num_batches

'''
Testing 

txt = "Artificial intelligence models rely heavily on the precise mathematical alignment of their loss functions during the training phase. When an algorithm processes input sequences, it calculates a numerical penalty based on the discrepancy between its predicted token distribution and the actual target text. Minimizing this error requires stable gradient updates and well-curated datasets. If the underlying data contains excessive noise or unexpected token patterns, the loss metrics may fluctuate unpredictably, signaling potential convergence issues. Therefore, engineers must rigorously evaluate loss behavior across diverse textual structures to ensure the network achieves optimal linguistic coherence and generalization."
tokenizer = Tokenizier()
cfg = {
    "vocab_size": tokenizer.encoder.n_vocab,
    "context_length": 1024,
    "emb_dim": 768,
    "num_heads": 12,
    "n_layers": 12,
    "dropout": 0.1,
    "qkv_bias": False
}
train_ratio = 0.6
split_idx = int(train_ratio * len(txt))
# make loaders 
Train_loader = Dataloader(txt[:split_idx],batch_size=4, max_size=10, num_workers=0, shuffle=True, drop_last=True, stride=5) 
val_loader = Dataloader(txt[split_idx:],batch_size=4, max_size=10, num_workers=0, shuffle=True, drop_last=True, stride=5)
model = Model(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target = calculate_loss(Train_loader, model, device, num_batches=None)
val = calculate_loss(val_loader, model, device, num_batches=None)
print(f"Train Loss: {target}")
print(f"Input Loss: {val}")
'''