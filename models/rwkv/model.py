########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
import torch.nn as nn
from torch.nn import functional as F

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method


def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out


########################################################################################################


class LayerNorm(MyModule):
    def __init__(self, embd, w):
        super().__init__()
        self.embd = embd
        self.w = w

    @MyFunction
    def forward(self, x, w):
        return F.layer_norm(x, (self.n_embd,), weight=w.weight, bias=w.bias)


class TimeMixing(MyModule):
    def __init__(self, embd, w, n_head, head_size):
        super().__init__()
        self.embd = embd
        self.w = w
        self.n_head = n_head
        self.head_size = head_size
        self.state_i1 = torch.zeros(embd.n_embd)
        self.state = torch.zeros(n_head, embd.n_embd)

    @MyFunction
    def time_mixing(
        self,
        x,
    ):
        H = self.n_head
        S = self.head_size
        w = self.w

        sx = self.state_i1 - x
        self.state_i1 = x
        xxx = x + sx * w.time_maa_x
        xxx = torch.tanh(xxx @ w.time_maa_w1).view(5, 1, -1)
        xxx = torch.bmm(xxx, w.time_maa_w2).view(5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + sx * (w.time_maa_w + mw)
        xk = x + sx * (w.time_maa_k + mk)
        xv = x + sx * (w.time_maa_v + mv)
        xr = x + sx * (w.time_maa_r + mr)
        xg = x + sx * (w.time_maa_g + mg)

        w = (
            w.time_decay + (torch.tanh(xw @ w.time_decay_w1) @ w.time_decay_w2).float()
        ).view(H, S, 1)
        w = torch.exp(-torch.exp(w.float()))

        r = (w.receptance.weight @ xr).view(H, 1, S)
        k = (w.key.weight @ xk).view(H, S, 1)
        v = (w.value.weight @ xv).view(H, 1, S)
        g = F.silu(w.gate.weight @ xg)

        s = self.state.reshape(H, S, S)

        x = torch.zeros(H, S)
        a = k @ v
        x = r @ (w.time_faaaa * a + s)
        s = a + w * s

        self.state = s.reshape(S, -1)
        x = x.flatten()

        x = (
            F.group_norm(
                x.unsqueeze(0),
                num_groups=H,
                weight=w.ln_x.weight,
                bias=w.ln_x.bias,
                eps=64e-5,
            ).squeeze(0)
            * g
        )  # same as gn(x/8, eps=1e-5)
        return w.output.weight @ x


class ChannelMixing(MyModule):
    def __init__(self, embd, w):
        super().__init__()
        self.embd = embd
        self.w = w
        self.state = torch.zeros(self.n_embd)

    @MyFunction
    def forward(self, x):
        w = self.w

        sx = self.state - x
        xk = x + sx * w.time_maa_k
        xr = x + sx * w.time_maa_r
        self.state = x
        r = torch.sigmoid(w.receptance.weight @ xr)
        k = torch.square(torch.relu(w.key.weight @ xk))  # square relu, primer paper
        return r * (w.value.weight @ k)


class RWKV_RNN(MyModule):
    def __init__(self, embd, w, n_head, head_size):
        super().__init__()
        self.embd = embd
        self.w = w
        self.id = id
        self.time_mixing = TimeMixing(embd, w.att, n_head, head_size)
        self.channel_mixing = ChannelMixing(embd, w.ffn, n_head, head_size)
        self.layer_norm_1 = LayerNorm(embd, w.ln1)
        self.layer_norm_2 = LayerNorm(embd, w.ln2)

    @MyFunction
    def forward(self, x):
        x = x + self.time_mixing(self.layer_norm_1(x, self.w.ln1))
        x = x + self.channel_mixing(self.layer_norm_2(x, self.w.ln2))


class RWKV_LM(MyModule):
    def __init__(self, embd):
        super().__init__()
        self.embd = embd
        self.eval()  # set torch to inference mode

        w = torch.load(embd.MODEL_NAME + ".pth", map_location="cpu")

        for k in w.keys():
            w[k] = w[k].float()  # convert to f32 type
            if ".time_" in k:
                w[k] = w[k].squeeze()
            if ".time_faaaa" in k:
                w[k] = w[k].unsqueeze(-1)

        self.n_head = w["blocks.0.att.time_faaaa"].shape[0]
        self.head_size = w["blocks.0.ln1.weight"].shape[0] // self.n_head

        self.w = types.SimpleNamespace()  # set self.w from w
        self.w.blocks = {}
        for (
            k
        ) in (
            w.keys()
        ):  # example: "blocks.0.att.w.time_faaaa" => self.w.blocks[0].att.w.time_faaaa
            parts = k.split(".")
            last = parts.pop()
            here = self.w
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
            setattr(here, last, w[k])

        self.blocks = [
            RWKV_RNN(
                embd=embd,
                w=self.w.blocks[i],
                id=i,
                n_head=self.n_head,
                head_size=self.head_size,
            )
            for i in range(self.embd.n_layer)
        ]

        self.layer_norm_0 = LayerNorm(embd, w.blocks[0].ln0)
        self.layer_norm_out = LayerNorm(embd, w.blocks[0].ln_out)

    def forward(self, token):
        with torch.no_grad():
            x = self.w.emb.weight[token]
            x = self.layer_norm_0(x)
            for i in range(self.embd.n_layer):
                x = self.blocks[i](x)

            x = self.w.head.weight @ self.layer_norm_out(x)
            return x.float()

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

