import os
import gc
import torch
import torch.nn.functional as F
from pynvml import *
from rwkv.utils import PIPELINE, PIPELINE_ARGS   
from rwkv.model import RWKV

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

#判断设备#
if torch.cuda.is_available():
    device = torch.device("cuda") #调用hip设备(其实写cuda就是hip)
    print("device :hip or cuda")
else: 
    device = torch.device("cpu")

model_path = "weights/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth" ##模型路径(可修改)
model = RWKV(model=model_path, strategy='cuda bf16')  ##调整策略
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")  ##模型词库
ctx_limit = 35000
penalty_decay = 0.996
################################################dialogue######################################################
model_state = None
Assistant = "Molice"
user_name = "user"
out_last = 0
occurrence = {}
out_tokens = []
answer = ""
msg = ""
out_str = ""

def chat(
    ctx,
    token_count,
    temperature=1.0,
    top_p=0.3,
    presencePenalty=0.3,
    countPenalty=0.3,
):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = model_state
    ctx += user_name + ": " + ctx + "\n\n" + Assistant + ": "
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
            occurrence[xxx] *= penalty_decay
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield out_str.strip()
            out_last = i + 1
    del out
    del state
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str.strip()
