import torch
from tqdm import trange
import numpy as np
import sys
import math

# only generate a single batch/head of data, which makes file loading much faster
B = 1  # batch size
N = int(sys.argv[1])  # sequence length
D = int(sys.argv[2])  # head dimension
H = int(sys.argv[3])  # number of heads
use_beta = bool(int(sys.argv[4]))  # whether to use beta term

torch.random.manual_seed(42)
q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
v = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
beta = torch.randn((B, N, H), dtype=torch.bfloat16, device='cuda').requires_grad_() if use_beta else None
grad_output = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')

##########################################
### EXACT DELTANET COMPUTATION ###
state = torch.zeros((B, H, D), dtype=torch.bfloat16, device='cuda')
output = torch.zeros((B, H, N, D), dtype=torch.bfloat16, device='cuda')

# L2 normalize q and k
q_norm = torch.nn.functional.normalize(q, p=2, dim=-1)
k_norm = torch.nn.functional.normalize(k, p=2, dim=-1)

for t in range(N):
    qk = torch.sum(q_norm[:,:,t,:] * k_norm[:,:,t,:], dim=-1, keepdim=True)  # [B, H, 1]
    if use_beta:
        qk = qk * beta[:, t, :].unsqueeze(-1)  # [B, H, 1]
    state = state + qk * (v[:,:,t,:] - state)  # [B, H, D]
    output[:,:,t,:] = state

### EXACT DELTANET COMPUTATION ###
##########################################

# Get reference output using existing implementation
o, _ = chunk_delta_rule(
    q=q,
    k=k,
    v=v,
    beta=beta,
    initial_state=None,
    output_final_state=False,
    head_first=False,
    use_qk_l2norm_in_kernel=True
)

# Do backwards computation
o.backward(grad_output)

q_grad = q.grad
k_grad = k.grad
v_grad = v.grad
beta_grad = beta.grad if use_beta else None

print("--------------------------------------")
print("Q shape: ", q.shape)
print("K shape: ", k.shape) 
print("V shape: ", v.shape)
print("O shape: ", o.shape)
print("Q grad shape: ", q_grad.shape)
print("K grad shape: ", k_grad.shape)
print("V grad shape: ", v_grad.shape)
if use_beta:
    print("Beta shape: ", beta.shape)
    print("Beta grad shape: ", beta_grad.shape)
print("--------------------------------------")

print(f'Average magnitude of OUTPUT tensor: {o.abs().mean()}')
print(f'1/100 magnitude of OUTPUT tensor:   {o.abs().mean()/100}')
print(f'Average magnitude of Q_GRAD tensor: {q_grad.abs().mean()}')
print(f'1/100 magnitude of Q_GRAD tensor:   {q_grad.abs().mean()/100}')
print(f'Average magnitude of K_GRAD tensor: {k_grad.abs().mean()}')
print(f'1/100 magnitude of K_GRAD tensor:   {k_grad.abs().mean()/100}')
print(f'Average magnitude of V_GRAD tensor: {v_grad.abs().mean()}')
print(f'1/100 magnitude of V_GRAD tensor:   {v_grad.abs().mean()/100}')
if use_beta:
    print(f'Average magnitude of BETA tensor: {beta.abs().mean()}')
    print(f'Average magnitude of BETA_GRAD tensor: {beta_grad.abs().mean()}')
print("--------------------------------------")

# Compare reference and exact implementations
print(f'Max difference between implementations: {(o - output).abs().max()}')
print("--------------------------------------")

filename = f'delta_randn_{N}N_{D}D_{H}H'
if use_beta:
    filename += '_beta'
filename += '.txt'

with open(filename, 'w') as f:
    # Write in ThunderKittens tile-friendly format
    for tensor, name in [(q, 'Q'), (k, 'K'), (v, 'V'), (o, 'O')]:
        tensor = tensor.permute(0, 2, 1, 3)  # [B, N, H, D] format
        data = tensor.to(torch.float32).flatten().detach().cpu().numpy()
        for val in tqdm(data, desc=f'Writing {name}'):
            f.write(f'{val} ')


