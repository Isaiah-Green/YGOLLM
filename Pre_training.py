from tokenizer import Tokenizier
from Model import Model
from Text_Generation import To_token,To_ouput,Generate_output_and_print,Generate_output_highest
from Loss import calculate_loss, calc_loss_batch
from Loaders import Dataloader
import torch
import random
def Train_Model_on_input(model, train_loader, value_loader, optimizer, device, tokenizer, num_epochs, eval_iter, eval_freq,start_context):
    train_loss , val_loss,track_tokens_seen = [],[],[]
    tokens_seen,global_step = 0,-1
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step +=1
            '''
            Testing
            '''
            #evaluate
            if global_step % eval_freq == 0:
                train_losses, val_losses = evaluate_model(model, train_loader, value_loader,device, eval_iter)
                model.train()
                train_loss.append(train_losses)
                val_loss.append(val_losses)
                track_tokens_seen.append(tokens_seen)
                start_context = ["Hello , IM feeling a bit ", "Do you want to come with", "No way !, i cant beleive you", " Sorry i was in the "]
                t = random.randint(0,3)
                y = start_context[t].split()
                print(f"{epoch + 1} (Step {global_step:06d}):" f"Train loss {train_losses:.3f}, Val loss {val_losses:.3f}")
                for step in range(1, len(y) + 1):
                    part = " ".join(y[:step])
                    Generate_output_and_print(model,tokenizer,part,device)
                

'''
Testing 
'''
import numpy as np
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # sets model to evaluation mode (good pratice)
    with torch.no_grad():
        train_loss = calculate_loss(train_loader, model, device, num_batches=eval_iter)
        val_loss = calculate_loss(val_loader, model, device, num_batches=eval_iter)
        model.train() # puts back in training mode
        return train_loss, val_loss

'''
Testing 
'''
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
with open("LM/Auto_Complete_Train_Data.txt", "r", encoding="utf-8") as f:
    text_data = f.read()
train_ratio = 0.7
split = int(train_ratio * len(text_data))
train_data = text_data[:split]
val_data = text_data[split:]
eval_iter, eval_freq = 2 , 25
train_loader = Dataloader(train_data)
val_loader = Dataloader(val_data)

model = Model(cfg)
tokenizer = Tokenizier()
start_context = ["Hello , IM feeling a bit ", "Do you want to come with", "No way ! , i cant beleive you", "Sorry i was in the "]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005,weight_decay=0.1)
num_epochs = 10
t = random.randint(0,3)
train_loss, val_loss, tokens_seen = Train_Model_on_input(model,train_loader,val_loader,optimizer,device,tokenizer,num_epochs,eval_iter,eval_freq,start_context[t])
