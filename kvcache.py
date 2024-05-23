
import torch

import time

from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class KVcache_MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(KVcache_MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

        # kvcache init
        self.k_cache = None
        self.v_cache = None

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        print("input_Q:", q.shape)
        print("input_K:", k.shape)
        print("input_V:", v.shape)
        
        # k,v concat  kv of kvcache
        if self.k_cache == None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat((self.k_cache, k), dim = 2)
            self.v_cache = torch.cat((self.v_cache, v), dim = 2)
            k = self.k_cache
            v = self.v_cache

        print("input_Q:", q.shape)
        print("input_K:", k.shape)
        print("input_V:", v.shape)
        

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


d_model = 512
n_head = 8
model = KVcache_MultiHeadAttention(d_model, n_head)

# 创建一个长序列
sequence_length = 1000
batch_size = 1
d_model = 512
sequence = torch.randn(batch_size, sequence_length, d_model)

# 准备查询、键和值
q = sequence
k = sequence
v = sequence

##测试不使用缓存
'''start_time = time.time()
for i in range(sequence_length):
    mask = None
    if i > 0:
        mask = torch.triu(torch.ones(i, i), diagonal=1).bool().unsqueeze(0)
    out = model(q[:, i:i+1, :], k[:, i:i+1, :], v[:, i:i+1, :], mask)
torch.cuda.synchronize() if torch.cuda.is_available() else None
no_cache_time = time.time() - start_time'''

# 重置模型缓存
model.k_cache = None
model.v_cache = None

# 测试使用缓存
start_time = time.time()
for i in range(2):
    mask = None
    if i > 0:
        mask = torch.triu(torch.ones(i, i), diagonal=1).bool().unsqueeze(0)
    out = model.forward(q,k,v,mask=None)
torch.cuda.synchronize() if torch.cuda.is_available() else None
with_cache_time = time.time() - start_time

# 打印性能差异
#print(f"No cache time: {no_cache_time} seconds")
print(f"With cache time: {with_cache_time} seconds")

        
