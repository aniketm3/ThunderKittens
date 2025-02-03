import torch
from tqdm import trange
import numpy as np
import sys
import math

B = 1
N = int(sys.argv[1])
D = int(sys.argv[2])

H_QO = int(sys.argv[3])
H_KV = int(sys.argv[4])

causal = False

# Produces random tensors we will use for test
torch.random.manual_seed(42)
x = torch.randn((B, N, D * H_QO), dtype=torch.bfloat16, device='cuda').requires_grad_()
grad_output = torch.randn((B, N, D * H_QO), dtype=torch.bfloat16, device='cuda')

freqs_cis = torch.randn((N, D // 2), dtype=torch.bfloat16, device='cuda')

# MLA Forward Pass
compress_q_linear = torch.nn.Linear(D * H_QO, D * H_QO, bias=False, device='cuda', dtype=torch.bfloat16)
compress_kv_linear = torch.nn.Linear(D * H_KV, D * H_KV, bias=False, device='cuda', dtype=torch.bfloat16)
q_norm = torch.nn.LayerNorm(D * H_QO, device='cuda', dtype=torch.bfloat16)
kv_norm = torch.nn.LayerNorm(D * H_KV, device='cuda', dtype=torch.bfloat16)

decompress_q_nope = torch.nn.Linear(D * H_QO, D * H_QO, bias=False, device='cuda', dtype=torch.bfloat16)
decompress_q_rope = torch.nn.Linear(D * H_QO, D * H_QO, bias=False, device='cuda', dtype=torch.bfloat16)
decompress_k_nope = torch.nn.Linear(D * H_KV, D * H_KV, bias=False, device='cuda', dtype=torch.bfloat16)
decompress_v_linear = torch.nn.Linear(D * H_KV, D * H_KV, bias=False, device='cuda', dtype=torch.bfloat16)
k_rope_linear = torch.nn.Linear(D * H_KV, D * H_KV, bias=False, device='cuda', dtype=torch.bfloat16)

compressed_q = compress_q_linear(x)
norm_q = q_norm(compressed_q)
query_nope = decompress_q_nope(norm_q)
query_rope = decompress_q_rope(norm_q)

compressed_kv = compress_kv_linear(x)
norm_kv = kv_norm(compressed_kv)
key_nope = decompress_k_nope(norm_kv)
value = decompress_v_linear(norm_kv)

key_rope = k_rope_linear(x)

query_nope = query_nope.view(B, N, H_QO, D).transpose(1, 2)
query_rope = query_rope.view(B, N, H_QO, D).transpose(1, 2)

key_rope = key_rope.view(B, N, 1, D).transpose(1, 2)
key_nope = key_nope.view(B, N, H_QO, D).transpose(1, 2)

value = value.view(B, N, H_QO, D).transpose(1, 2)
value = value * (1 / math.sqrt(D))

# Apply rotary embeddings
q_rope, k_rope = query_rope, key_rope  # Replace with proper rotary embedding function

q_recombined = torch.empty((B, H_QO, N, D * 2), device=x.device)
k_recombined = torch.empty((B, H_QO, N, D * 2), device=x.device)

q_recombined[:, :, :, :D] = query_nope
q_recombined[:, :, :, D:] = q_rope

k_recombined[:, :, :, :D] = key_nope
k_recombined[:, :, :, D:] = k_rope

output = torch.nn.functional.scaled_dot_product_attention(q_recombined, k_recombined, value, is_causal=causal)

output = output.transpose(1, 2).contiguous().view(B, N, H_QO * D)

output.backward(grad_output)

q_grad = x.grad

print("--------------------------------------")
print("X shape: ", x.shape)
print("Output shape: ", output.shape)
print("X grad shape: ", q_grad.shape)
print("--------------------------------------")

# Save output in the same format
filename = f'randn_{N}N_{D}D_{H_QO}QO_{H_KV}KV_mla.txt'
with open(filename, 'w') as f:
    xf = x.to(torch.float32).flatten().detach().cpu().numpy()
    of = output.to(torch.float32).flatten().detach().cpu().numpy()
    qg_f = q_grad.to(torch.float32).flatten().detach().cpu().numpy()
    
    for val in xf:
        f.write(repr(float(val)) + ' ')
    for val in of:
        f.write(repr(float(val)) + ' ')
    for val in qg_f:
        f.write(repr(float(val)) + ' ')
