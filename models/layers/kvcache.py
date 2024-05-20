from torch import nn


class KVcache(nn.Module):
    def__init__(self,d_model,n_head):
    super(KVcache,self).__init__()
    self.n_head=n_head
    self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)


    def forward(self,tgt,tgt_mask=None):

        if self.k_cache is None or self.v_cache is None:
        # 初始化缓存
        self.k_cache = tgt.new_zeros(0, tgt.size(1), self.self_attn.k_v_dim)
        self.v_cache = tgt.new_zeros(0, tgt.size(1), self.self_attn.k_v_dim)

        # 扩展KV缓存以包含当前时间步的信息
        new_k = tgt.new_zeros(tgt.size(0), 1, self.self_attn.k_v_dim)
        new_v = tgt.new_zeros(tgt.size(0), 1, self.self_attn.k_v_dim)
        new_k.copy_(tgt.view(tgt.size(0), 1, -1))
        new_v.copy_(tgt.view(tgt.size(0), 1, -1))

        # 更新缓存
        self.k_cache = torch.cat([self.k_cache, new_k], dim=1)
        self.v_cache = torch.cat([self.v_cache, new_v], dim=1)

        

