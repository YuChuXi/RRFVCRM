import os
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0 ' # if '1' then use CUDA kernel for seq mode (much faster)
# make sure cuda dir is in the same level as modeling_rwkv.py
from .modeling_rwkv import RWKV

import gc
import gradio as gr
import base64
from io import BytesIO
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ctx_limit = 3500
title = 'ViusualRWKV-v5'
weights_path = os.path.abspath("..")
rwkv_remote_path = "weights/rwkv1b5-vitl336p14-577token_mix665k_rwkv.pth"
vision_remote_path = "weights/rwkv1b5-vitl336p14-577token_mix665k_visual.pth"
vision_tower_name = 'openai/clip-vit-large-patch14-336'

model_path = rwkv_remote_path
model = RWKV(model=model_path, strategy='cuda fp16')
from rwkv.utils import PIPELINE, PIPELINE_ARGS
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

##########################################################################
from .modeling_vision import VisionEncoder, VisionEncoderConfig
config = VisionEncoderConfig(n_embd=model.args.n_embd, 
                             vision_tower_name=vision_tower_name, 
                             grid_size=-1)
visual_encoder = VisionEncoder(config)
vision_local_path = vision_remote_path
vision_state_dict = torch.load(vision_local_path, map_location='cpu')
visual_encoder.load_state_dict(vision_state_dict, strict=False)
image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
visual_encoder = visual_encoder.to(device)
##########################################################################
def generate_prompt(instruction):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    return f"\n{instruction}\n\nAssistant:"

def generate(
    ctx,
    image_state,
    token_count=200,
    temperature=0.2,
    top_p=0.3,
    presencePenalty = 0.0,
    countPenalty = 1.0,
):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                    alpha_frequency = countPenalty,
                    alpha_presence = presencePenalty,
                    token_ban = [], # ban the generation of some tokens
                    token_stop = [0, 261]) # stop generation whenever you see any token here
    ctx = ctx.strip()
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    for i in range(int(token_count)):
        if i == 0:
            input_ids = pipeline.encode(ctx)[-ctx_limit:]
            out, state = model.forward(tokens=input_ids, state=image_state)
        else:
            input_ids = [token]
            out, state = model.forward(tokens=input_ids, state=state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
            occurrence[xxx] *= 0.996        
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


##########################################################################

def pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format as needed (JPEG, PNG, etc.)
    # Encodes the image data into base64 format as a bytes object
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_image

image_cache = {}
ln0_weight = model.w['blocks.0.ln0.weight'].to(torch.float32).to(device)
ln0_bias = model.w['blocks.0.ln0.bias'].to(torch.float32).to(device)
def compute_image_state(image):
    base64_image = pil_image_to_base64(image)
    if base64_image in image_cache:
        image_state = image_cache[base64_image]
    else:
        image = image_processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values']
        image = image.to(device)
        image_features = visual_encoder.encode_images(image.unsqueeze(0)).squeeze(0) # [L, D]
        # apply layer norm to image feature, very important
        image_features = F.layer_norm(image_features, 
                                    (image_features.shape[-1],), 
                                    weight=ln0_weight, 
                                    bias=ln0_bias)
        _, image_state = model.forward(embs=image_features, state=None)
        image_cache[base64_image] = image_state
    return image_state

def chatbot_v(image, question):
    if image is None:
        yield "Please upload an image."
        return
    image_state = compute_image_state(image)
    input_text = generate_prompt(question)
    for output in generate(input_text, image_state):
        yield output