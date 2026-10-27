import torch
from tokenizer import Tokenizier
from Model import Model
import sys
sys.stdout.reconfigure(encoding='utf-8')
def To_token(tokenizer, input):
    if input[:3] == "CFUI":
        token = tokenizer.special_encode(input)
    else:
        token = tokenizer.encode(input)
    token = torch.tensor(token).unsqueeze(0)
    return token
        
def To_ouput(tokenizer , tokens):
    decode = tokens.squeeze(0)
    return tokenizer.decodes(decode.tolist())

def Generate_output_highest(model,idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]# gets last row
        probability = torch.softmax(logits, dim=-1)
        # use argmax to get highest index value
        idx_next = torch.argmax(probability, dim=-1, keepdim=True)
        top_k_prob, top_k_index = torch.topk(probability, k=3, dim=-1)
        idx = torch.cat((idx, idx_next), dim=-1)

    return idx, top_k_index

def Generate_output_and_print(model, tokenizer, start_context, device):
    model.eval()
    context_size = model.pos_embedding.weight.shape[0]
    encoded = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0)
    with torch.no_grad():
        token_ids,top_k = Generate_output_highest(model,idx=encoded,max_new_tokens=256, context_size=context_size)
    #top_k = top_k.unsqueeze(0)
    decoded_top_k = To_ouput(tokenizer, top_k[0])
    decode = To_ouput(tokenizer, token_ids)
    print(f"Output Message: {decode}") #decode returns a string
    return decode, decoded_top_k
'''
Testing

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
model = Model(cfg)
tokenizer = Tokenizier()
start_context = "The こんにちは is Nice"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
to_token = To_token(tokenizer, start_context)
to_output = To_ouput(tokenizer,to_token)
print(to_token)
print(to_output)
final = Generate_output_and_print(model, tokenizer, start_context, device)
print(final)
'''