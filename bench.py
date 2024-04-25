import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 1
n_head = 1
seq_len = 128
head_embd = 32

#incrementing_seq_q = torch.linspace(0,1,steps=head_embd*1).unsqueeze(0).repeat_interleave(seq_len, dim=0)
#incrementing_seq_k = torch.linspace(0,0.1,steps=head_embd*1).unsqueeze(0).repeat_interleave(seq_len, dim=0)
#incrementing_seq_v= torch.linspace(0,11,steps=head_embd*1).unsqueeze(0).repeat_interleave(seq_len, dim=0)

#q = incrementing_seq_q.unsqueeze(0).repeat(batch_size, n_head, 1, 1).cuda()
#k = incrementing_seq_k.unsqueeze(0).repeat(batch_size, n_head, 1, 1).cuda()
#v = incrementing_seq_v.unsqueeze(0).repeat(batch_size, n_head, 1, 1).cuda()  

#q = q.float()  # 或者 q.double()
#k = k.float()  # 或者 k.double()
#v = v.float()

#first_half = torch.arange(head_embd * 32).view(32, 32)

"""first_half = torch.randint(0, 101, (32, 32))
full_matrix = torch.cat((first_half, first_half), dim=0)
final_tensor = full_matrix.unsqueeze(0).unsqueeze(0).repeat(batch_size,n_head,1,1)
final_tensor = final_tensor.float()


first_half_k = torch.randint(0, 101, (32, 32))
full_matrix_k = torch.cat((first_half_k, first_half_k), dim=0)
final_tensor_k = full_matrix_k.unsqueeze(0).unsqueeze(0).repeat(batch_size,n_head,1,1)
final_tensor_k = final_tensor_k.float()


first_half_v= torch.randint(0, 101, (32, 32))
full_matrix_v= torch.cat((first_half_v, first_half_v), dim=0)
final_tensor_v = full_matrix_v.unsqueeze(0).unsqueeze(0).repeat(batch_size,n_head,1,1)
final_tensor_v = final_tensor_v.float()"""



"""first_half = torch.rand(32, 32, dtype=torch.float32)
first_half = first_half * 2 - 1
full_matrix = torch.cat((first_half, first_half), dim=0)
final_tensor = full_matrix.unsqueeze(0).unsqueeze(0).repeat(batch_size,n_head,1,1)
final_tensor = final_tensor.float()

first_half_k = torch.rand(32, 32, dtype=torch.float32)
first_half_k = first_half_k * 2 - 1
full_matrix_k = torch.cat((first_half_k, first_half_k), dim=0)
final_tensor_k = full_matrix_k.unsqueeze(0).unsqueeze(0).repeat(batch_size,n_head,1,1)
final_tensor_k = final_tensor_k.float()

first_half_v= torch.rand(32, 32, dtype=torch.float32)
first_half_v = first_half_v * 2 - 1
full_matrix_v= torch.cat((first_half_v, first_half_v), dim=0)
final_tensor_v = full_matrix_v.unsqueeze(0).unsqueeze(0).repeat(batch_size,n_head,1,1)
final_tensor_v = final_tensor_v.float()

q = final_tensor.cuda()
k = final_tensor_k.cuda()
v = final_tensor_v.cuda()"""

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()


#q = torch.ones(batch_size, n_head, seq_len, head_embd).cuda()
#k = torch.ones(batch_size, n_head, seq_len, head_embd).cuda()
#v = torch.ones(batch_size, n_head, seq_len, head_embd).cuda()

print('=== profiling manual attention ===')

def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
#print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v)
#print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('attn values sanity check:', torch.allclose(minimal_result, manual_result, rtol=1e-07, atol=0))


#torch.set_printoptions(threshold=10_000)

print(minimal_result)

print(manual_result)

