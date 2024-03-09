import torch,os

os.environ["RWKV_JIT_ON"] = "0"
os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_CTXLEN"] = "4096"

from rwkv import RWKV_LM, RWKV_TOKENIZER, sample_logits


if __name__ == "__main__":
    W = torch.load("weights/pretrained/RWKV-x060-World-1B6-v2-20240208-ctx4096.pth",map_location="cpu")
    model = RWKV_LM(n_layer=24,n_embd=2048,W=W)
    tokenizer = RWKV_TOKENIZER("models/rwkv/rwkv_vocab_v20230424.txt")
    prompt = input(">>>")

    for i in tokenizer.encode(prompt):
        logit = model.forward(i)
    i = sample_logits(logit)
    for n in range(128):
        print(tokenizer.decode([i]),end="")
        logit = model.forward(i)
        i = sample_logits(logit)
    print(tokenizer.decode([i]),end="")
