import os
from rwkv6.src.model import RWKV
import gc
import torch
import torch.nn.functional as F
from pynvml import *
from rwkv.utils import PIPELINE, PIPELINE_ARGS   

#判断设备#
if torch.cuda.is_available():
    device = torch.device("cuda") #调用hip设备(其实写cuda就是hip)
    print("device :hip or cuda")
else: 
    device = torch.device("cpu")

model_path = "/media/alic-li/WDdata03/RWKV-model/RWKV-v4-14b" ##模型路径(可修改)
model = RWKV(model=model_path, strategy='cuda fp16')  ##调整策略
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")  ##模型词库
ctx_limit = 35000

################################################dialogue######################################################
model_state = None
Assistant = "Molice"
out_last = 0
occurrence = {}
out_tokens = []
answer = ""
msg = ""
out_str = ""

def chat(
    user_name,
    ctx,
    token_count,
    temperature,
    top_p,
    presencePenalty,
    countPenalty,
):
    global model_state, Assistant, occurrence, answer, msg, out_last, out_str, out_tokens
    if user_name == None:
        user_name = "Bob"

    if msg != "" :
        pass
    #elif os.path.exists("./model-data/" + user_name + ".txt"):      ##启用历史聊天记录分析
        #msg = open("./model-data/" + user_name + ".txt").read()
    elif os.path.exists("./model-data/" + user_name + ".txt"):
        msg = ""
    else:
        msg = open("./model-data/" + user_name + ".txt")   ##新建聊天记录
        msg =""  ##清空当前聊天记录
    
    msg += user_name + ": " + ctx + "\n\n" + Assistant + ": "
    
    if os.path.exists("./model-data/" + user_name + ".pth"):   #加载历史状态
        model_state = torch.load("./model-data/" + user_name + ".pth", map_location=device)
    else:
        model_state = None              ##新建状态
    
    tokens = pipeline.encode(msg)
    out, model_state = model.forward(tokens, model_state)
    
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    
    for i in range(int(token_count)):
        for n in occurrence:
            out[n] -= args.alpha_presence + occurrence[n] * args.alpha_frequency # repetition penalty
        out[0] -= 1e10  # disable END_OF_TEXT
        
        token = pipeline.sample_logits(out, temperature, top_p)
        
        for xxx in occurrence:
            occurrence[xxx] *= 0.99
        occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
        out, model_state = model.forward([token], model_state)
        out_tokens += [token]
        answer += pipeline.decode([token])
        if "\n\n" in answer:
            yield answer.strip()
            break
    if "\n\n" in answer:
            yield answer.strip()
    msg += answer
    text =  user_name + ": " + ctx + "\n\n" + Assistant + ": " + answer
    text_file = open("./model-data/" + user_name + ".txt", "a")
    text_file.write(text)
    torch.save(model_state,"./model-data/" + user_name + ".pth")
    answer = ""
    model_state = None
    gc.collect()    
    torch.cuda.empty_cache()   
    return user_name, model_state, msg