from gpt_download import download_and_load_gpt2
import numpy as np
import torch
from Model import Model
#settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(model, params):
    model.pos_embedding.weight = assign(model.pos_embedding.weight, params["wpe"])
    model.embedding.weight     = assign(model.embedding.weight,     params["wte"])

    for b in range(len(params["blocks"])):
        blk = params["blocks"][b]
        t = model.Transformer[b]

        # Attention: GPT-2 stores Q, K, V in one matrix, so split it in 3
        q_w, k_w, v_w = np.split(blk["attn"]["c_attn"]["w"], 3, axis=-1)
        t.attention.weight_q.weight = assign(t.attention.weight_q.weight, q_w.T)
        t.attention.weight_k.weight = assign(t.attention.weight_k.weight, k_w.T)
        t.attention.weight_v.weight = assign(t.attention.weight_v.weight, v_w.T)

        q_b, k_b, v_b = np.split(blk["attn"]["c_attn"]["b"], 3, axis=-1)
        t.attention.weight_q.bias = assign(t.attention.weight_q.bias, q_b)
        t.attention.weight_k.bias = assign(t.attention.weight_k.bias, k_b)
        t.attention.weight_v.bias = assign(t.attention.weight_v.bias, v_b)

        t.attention.out_projection.weight = assign(t.attention.out_projection.weight, blk["attn"]["c_proj"]["w"].T)
        t.attention.out_projection.bias   = assign(t.attention.out_projection.bias,   blk["attn"]["c_proj"]["b"])

        # Feed-forward
        t.feed.layer[0].weight = assign(t.feed.layer[0].weight, blk["mlp"]["c_fc"]["w"].T)
        t.feed.layer[0].bias   = assign(t.feed.layer[0].bias,   blk["mlp"]["c_fc"]["b"])
        t.feed.layer[2].weight = assign(t.feed.layer[2].weight, blk["mlp"]["c_proj"]["w"].T)
        t.feed.layer[2].bias   = assign(t.feed.layer[2].bias,   blk["mlp"]["c_proj"]["b"])

        # LayerNorms (nn.LayerNorm uses .weight / .bias)
        t.norm1.weight = assign(t.norm1.weight, blk["ln_1"]["g"])
        t.norm1.bias   = assign(t.norm1.bias,   blk["ln_1"]["b"])
        t.norm2.weight = assign(t.norm2.weight, blk["ln_2"]["g"])
        t.norm2.bias   = assign(t.norm2.bias,   blk["ln_2"]["b"])

    model.Layer_Norm.weight = assign(model.Layer_Norm.weight, params["g"])
    model.Layer_Norm.bias   = assign(model.Layer_Norm.bias,   params["b"])
    model.logits.weight     = assign(model.logits.weight,     params["wte"])
