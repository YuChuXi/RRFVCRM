import types, torch
import torch.nn as nn
from torch.nn import functional as F
from .rnn import MyFunction, MyModule, LayerNorm, RWKV_RNN

class RWKV_LM(MyModule):
    def __init__(self, n_layer, n_embd, W):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.eval()  # set torch to inference mode

        for k in W.keys():
            W[k] = W[k].bfloat16()  # convert to bf16 type
            if ".time_" in k:
                W[k] = W[k].squeeze()
            if ".time_faaaa" in k:
                W[k] = W[k].unsqueeze(-1)

        self.n_head = W["blocks.0.att.time_faaaa"].shape[0]
        self.head_size = W["blocks.0.ln1.weight"].shape[0] // self.n_head

        self.W = types.SimpleNamespace()  # set self.w from w
        self.W.blocks = {}
        for (
            k
        ) in (
            W.keys()
        ):  # example: "blocks.0.att.W.time_faaaa" => self.W.blocks[0].att.W.time_faaaa
            parts = k.split(".")
            last = parts.pop()
            here = self.W
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here:
                        here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p):
                        setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, W[k])
        del W

        self.blocks = [
            RWKV_RNN(
                id=i,
                n_embd=n_embd,
                n_head=self.n_head,
                head_size=self.head_size,
                W=self.W.blocks[i],
            )
            for i in range(n_layer)
        ]

        self.layer_norm_0 = LayerNorm(n_embd, self.W.blocks[0].ln0)
        self.layer_norm_out = LayerNorm(n_embd, self.W.ln_out)

    @MyFunction
    def forward(self, token):
        with torch.no_grad():
            x = self.W.emb.weight[token]
            x = self.layer_norm_0(x)
            for i in range(self.n_layer):
                x = self.blocks[i](x)

            x = self.W.head.weight @ self.layer_norm_out(x)
            return x.bfloat16()

    def get_state(self):
        return [
            {
                "time_mixing_state": b.time_mixing.state,
                "time_mixing_state_i1": b.time_mixing.state_i1,
                "channel_mixing.state": b.channel_mixing.state,
            }
            for b in self.blocks
        ]

    def set_state(self, state):
        for b in self.blocks:
            id = b.id
            b.time_mixing.state = state[id]["time_mixing_state"]
            b.time_mixing.state_i1 = state[id]["time_mixing_state_i1"]
            b.channel_mixing.state = state[id]["channel_mixing.state"]

