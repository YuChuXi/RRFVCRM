import os
import gc
import torch
import torch.nn.functional as F
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
model = RWKV(model=model_path, strategy='cuda fp16')  ##调整策略
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
        if "\n\n" in tmp :
            break
    del out
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str.strip()
    
########################## text rwkv ################################################################
def evaluate(
    ctx,
    token_count=200,
    temperature=1.0,
    top_p=0.7,
    presencePenalty = 0.1,
    countPenalty = 0.1,
):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    ctx = ctx.strip()
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(int(token_count)):
        input_ids = pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token]
        out, state = model.forward(tokens=input_ids, state=state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
            occurrence[xxx] *= 0.996
            
        ttt = pipeline.decode([token])
        www = 1
        if ttt in ' \t0123456789':
            www = 0
        #elif ttt in '\r\n,.;?!"\':+-*/=#@$%^&_`~|<>\\()[]{}，。；“”：？！（）【】':
        #    www = 0.5
        if token not in occurrence:
            occurrence[token] = www
        else:
            occurrence[token] += www
            
        anser = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in anser:
            out_str += anser
            yield out_str.strip()
            out_last = i + 1

    #gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    #timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #print(f'{timestamp} - vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')       #显示GPU占用，希望可以找到调用RadeonGPU占用的库
    del out
    del state
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str.strip()