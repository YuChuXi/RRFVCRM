import torch

from rwkv import RWKV_LM, WorldTokenizer, sample_logits


if __name__ == "__main__":
    torch.set_num_threads(8)
    W = torch.load("weights/pretrained/RWKV-x060-World-1B6-v2-20240208-ctx4096.pth",map_location="cpu")
    model = RWKV_LM(n_layer=24,n_embd=2048,W=W)
    tokenizer = WorldTokenizer("models/rwkv/rwkv_vocab_v20230424.txt")
    prompt = input(">>>")

    for i in tokenizer.encode(prompt):
        logit = model.forward(i)
    i = sample_logits(logit)
    for n in range(128):
        print(tokenizer.decode([i]),end="")
        logit = model.forward(i)
        i = sample_logits(logit)
    print(tokenizer.decode([i]),end="")
