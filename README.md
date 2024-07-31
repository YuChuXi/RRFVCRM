# ![RRFVCM](assets/logo.png) Welcome to RRVtuber
![Language](https://img.shields.io/badge/language-python-brightgreen) ![Documentation](https://img.shields.io/badge/documentation-yes-brightgreen)

[中文文档](./README-ch.md)

## Introduction
An AI for generating voice and actions based on the RWKV model architecture

### Motivation

### Conventions
- Commands in the documentation are to be executed in the project's root directory unless otherwise specified
- `python` and `python3` are the same

# 🛠 Preparation

### Setting Up the Environment
1. Install [Python](https://python.org)
2. Install CUDA/ROCm and the corresponding version of PyTorch
3. Install the required libraries
```sh
pip install -r requirements.txt
```
4. If you are using an AMD GPU, add the following commands to `~/.bashrc` (using gfx1100 as an example, you can find the specific model by running `rocminfo`)
```sh
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```
5. Run the following commands
```sh
sudo usermod -aG render $USERNAME 
sudo usermod -aG video $USERNAME 
```
#### If you are an AMD user and want to add cuda operator parallelization, it will be a bit trouble. You need to modify the rwkv standard library, and it may not work.
#### Really want to RUN？ Well ~~~
```sh
cd ~/.local/lib/python3.10/site-packages/rwkv
vim ./model.py
```
- Change lines 37, 46, 472, 505 from `extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"]` to `extra_cuda_cflags=["-O3", "--hipstdpar", "-xhip"]`
- Globally search for `os.environ["RWKV_CUDA_ON"] = '0'` and change it to `os.environ["RWKV_CUDA_ON"] = '1'`
```sh
python webui.py
```
#### Good luck!

##### You will find a hip directory under `~/.local/lib/python3.10/site-packages/rwkv`, which contains the converted CUDA parallel operators

##### Failed? Globally search for `os.environ["RWKV_CUDA_ON"] = '1'` and change it to `os.environ["RWKV_CUDA_ON"] = '0'`

### 📥 Download Pre-trained Weights
Pre-trained weights are stored in `./weights/`
- RWKV-LM [RWKV-x060-World-1B6-v2-20240208-ctx4096.pth](https://huggingface.co/BlinkDL/rwkv-6-world/blob/main/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth)
- Visaul-RWKV-LM [rwkv1b5-vitl336p14-577token_mix665k_rwkv.pth](https://huggingface.co/howard-hou/visualrwkv-5/blob/main/rwkv1b5-vitl336p14-577token_mix665k_rwkv.pth)
- Visaul-RWKV [rwkv1b5-vitl336p14-577token_mix665k_visual.pth](https://huggingface.co/howard-hou/visualrwkv-5/blob/main/rwkv1b5-vitl336p14-577token_mix665k_visual.pth)
- Bert [s1bert.ckpt](https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s1bert25hz-2kh-longer-epoch%3D68e-step%3D50232.ckpt)
- HuBert [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)
- RMVPE [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt)

### 📝 Modify Pre-trained Weights Path
- Line 19 in `./models/rwkv6/dialogue.py`
- Line 19 in `./models/rwkv6/continuation.py`
- Line 17 in `./models/music/run.py`
- Line 11 in `./models/language_test.py`
- line 19 and line 20 in `./models/visualRWKV/app/app_gpu.py`
# 🧪 Verification
- Execute
```sh 
python models/language_test.py
``` 
- If it interacts normally, the preparation work is correct

# 🚀 Quick Run Language Model (It is Available now!)
```sh
python webui.py
```
# 👀  Quick Run Visual-RWKV Model （It is Available now!）
```sh
python webui.py
```
#### Adjust the model running strategy in line 19 of `models/rwkv6/dialogue.py`, default "cuda fp16"
#### Adjust the Visual-RWKV model running strategy in line 24 of`models/visualRWKV/app/app_gpu.py` default "cuda fp16"
Alic is a noob in the DeepLearning ，but it's could be running

## 📂 Project Structure

## 🧠 Training (It is not Available now!)- Wait for YuChuXi, She is a lazy little fox
- Training requires [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace/releases) to extract facial features. After installation, configure the path in `config/openseeface.json`
- For some datasets, automatic speech annotation may be required [DeepSpeech](https://github.com/mozilla/DeepSpeech)

### 📦 Prepare Data
You can prepare the data yourself or refer to the following datasets
- Speech, Text [Mozilla Common Voice](https://commonvoice.mozilla.org/zh-CN)
- Speech, Face [CelebV-Text](https://github.com/celebv-text/CelebV-Text)

### ⚙️ Data Preprocessing
- Video or audio slicing (default 25FPS * 40s per slice, corresponding to non-language model 25FPS * 1024CTX) `python`
- Extract hubert and f0 `python`
- Extract facial features from video `python`

### 🎶 Train T2F0
- Wait for YuChuXi,She is a lazy little fox
### 🎶 Train TF02M
- Wait for YuChuXi,She is a lazy little fox

# 🌟 Extensions
Try rwkv-music-demo
--
- Prepare the model (choose MIDI-model)
https://huggingface.co/BlinkDL/rwkv-5-music/tree/main
```sh
cd ./models/music
python ./run.py
```
- The model path is on line 17 of `run.py`. If it does not run properly, change line 22 from "strategy='cuda fp32'" to "strategy='cpu fp32'"

State Tuning
--
Refer to https://github.com/JL-er/RWKV-PEFT

rwkv-language-test
--
- Go to `./models/rwkv/`
- Run `python language_test.py`

## ❓ Having Issues?
- If yuo can can't run ```webui.py```In most cases, the command line terminal may not be able to connect to the Huggingface website. Please try using a proxy and set the proxy in the command line terminal.
```sh
export https_proxy=http://127.0.0.1:[port]
export http_proxy=http://127.0.0.1:[port]
```
- parselmouth installation failed: temporarily downgrade `setuptools` to below 58.0

## Other

### Future Directions

### Acknowledgements
- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- [Visual-RWKV](https://github.com/howard-hou/VisualRWKV)
- [SoftVC VITS Voice Conversion](https://github.com/justinjohn0306/so-vits-svc-4.0/tree/4.0-v2)
- [GPT-SoVITS-WebUI](https://github.com/RVC-Boss/GPT-SoVITS)
- [RMVPE](https://github.com/Dream-High/RMVPE)
- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace)
- [DeepSpeech](https://github.com/mozilla/DeepSpeech)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/zh-CN)
- [CelebV-Text](https://github.com/celebv-text/CelebV-Text)
- Compute cards from [AMD](https://amd.com)
