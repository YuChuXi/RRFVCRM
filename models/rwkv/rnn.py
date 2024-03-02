########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
# Object By - https://github.com/yuchuxi
########################################################################################################

import torch
from torch import nn
from torch.nn import functional as F


MyModule = nn.Module  # torch.jit.ScriptModule
MyFunction = lambda x: x  # torch.jit.script_method


########################################################################################################


class LayerNorm(MyModule):
    def __init__(self, n_embd, W):
        super().__init__()
        self.n_embd = n_embd
        self.W = W

    @MyFunction
    def forward(self, x):
        W = self.W
        return F.layer_norm(x, (self.n_embd,), weight=W.weight, bias=W.bias)


class TimeMixing(MyModule):
    def __init__(self, n_embd, n_head, head_size, W):
        super().__init__()
        self.n_embd = n_embd
        self.W = W
        self.n_head = n_head
        self.head_size = head_size
        self.state_i1 = torch.zeros(n_embd)
        self.state = torch.zeros(head_size, n_embd)

    @MyFunction
    def forward(
        self,
        x,
    ):
        H = self.n_head
        S = self.head_size
        W = self.W

        sx = self.state_i1 - x
        self.state_i1 = x
        xxx = x + sx * W.time_maa_x
        xxx = torch.tanh(xxx @ W.time_maa_w1).view(5, 1, -1)
        xxx = torch.bmm(xxx, W.time_maa_w2).view(5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + sx * (W.time_maa_w + mw)
        xk = x + sx * (W.time_maa_k + mk)
        xv = x + sx * (W.time_maa_v + mv)
        xr = x + sx * (W.time_maa_r + mr)
        xg = x + sx * (W.time_maa_g + mg)

        ww = (
            W.time_decay + (torch.tanh(xw @ W.time_decay_w1) @ W.time_decay_w2).bfloat16()
        ).view(H, S, 1)
        ww = torch.exp(-torch.exp(ww.bfloat16()))

        r = (W.receptance.weight @ xr).view(H, 1, S)
        k = (W.key.weight @ xk).view(H, S, 1)
        v = (W.value.weight @ xv).view(H, 1, S)
        g = F.silu(W.gate.weight @ xg)

        s = self.state.reshape(H, S, S)

        x = torch.zeros(H, S)
        a = k @ v
        x = r @ (W.time_faaaa * a + s)
        s = a + ww * s

        self.state = s.reshape(S, -1)
        x = x.flatten()

        x = (
            F.group_norm(
                x.unsqueeze(0),
                num_groups=H,
                weight=W.ln_x.weight,
                bias=W.ln_x.bias,
                eps=64e-5,
            ).squeeze(0)
            * g
        )  # same as gn(x/8, eps=1e-5)
        return W.output.weight @ x


class ChannelMixing(MyModule):
    def __init__(self, n_embd, W):
        super().__init__()
        self.n_embd = n_embd
        self.W = W
        self.state = torch.zeros(self.n_embd)

    @MyFunction
    def forward(self, x):
        W = self.W

        sx = self.state - x
        xk = x + sx * W.time_maa_k
        xr = x + sx * W.time_maa_r
        self.state = x
        r = torch.sigmoid(W.receptance.weight @ xr)
        k = torch.square(torch.relu(W.key.weight @ xk))  # square relu, primer paper
        return r * (W.value.weight @ k)


class RWKV_RNN(MyModule):
    def __init__(self, id, n_embd, n_head, head_size, W):
        super().__init__()
        self.n_embd = n_embd
        self.W = W
        self.id = id
        self.time_mixing = TimeMixing(n_embd, n_head, head_size, W.att)
        self.channel_mixing = ChannelMixing(n_embd, W.ffn)
        self.layer_norm_1 = LayerNorm(n_embd, W.ln1)
        self.layer_norm_2 = LayerNorm(n_embd, W.ln2)

    @MyFunction
    def forward(self, x):
        x = x + self.time_mixing(self.layer_norm_1(x))
        x = x + self.channel_mixing(self.layer_norm_2(x))
        return x.bfloat16()