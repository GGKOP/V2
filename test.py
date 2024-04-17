"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import time

import torch

from data import *
from models.model.transformer import Transformer


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        m.weight.data.fill_(0.001)
    if hasattr(m, "bias"):
        m.bias.data.fill_(0)


if __name__ == "__main__":
    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_sos_idx=trg_sos_idx,
        d_model=d_model,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=max_len,
        ffn_hidden=ffn_hidden,
        n_head=n_heads,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device,
    ).to(device)
    model.apply(initialize_weights)
    model.train()
    for batch in train_iter:
        src = batch.src
        trg = batch.trg

        start_time = time.time()
        output = model(src, trg[:, :-1])
        print("One step complete. %.2fs" % (time.time() - start_time))
        break
