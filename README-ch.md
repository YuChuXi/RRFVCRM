# RRVtuber
# ![RRVtuber](assets/logo.png)
## 前言
一个基于 RWKV 模型架构具有视觉的语音、动作生成 AI

### 初衷
本项目可灵活应用于AI虚拟主播或实体机器人的各类本地部署，具有未来节省算力、功耗的特点。本AI项目具备视觉情绪表达、动作生成等功能，目前正在完善各项功能。并且已经具备视觉化。
### 约定
- 文档中的指令均位于项目根目录执行，除非特别说明
- `python` 和 `python3` 是一样的

# 🛠 准备工作

### 配置环境
1. 安装 [Python](https://python.org)
2. 安装 CUDA/ROCm 和对应版本的 PyTorch
3. 安装依赖库
```sh
pip install -r requirements.txt
```
4. 如果你用的是 AMD 显卡，请在 `~/.bashrc` 中加入以下指令（以 gfx1100 为例，具体型号可通过运行 `rocminfo` 查看）
```sh
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```
5. 执行以下命令
```sh
sudo usermod -aG render $USERNAME 
sudo usermod -aG video $USERNAME 
```
#### 如果你是 AMD 用户，想要加上cuda算子并行化，那有点麻烦了，需要改rwkv标准库，还不一定跑得起来
#### 真的要跑？好吧~~~
```sh
cd ~/.local/lib/python3.10/site-packages/rwkv
vim ./model.py
```
- 修改第 37 行、46 行、472 行、505 行 `extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"]` 为 `extra_cuda_cflags=["-O3", "--hipstdpar", "-xhip"]`
- 全局搜索 `os.environ["RWKV_CUDA_ON"] = '0'` 改为 `os.environ["RWKV_CUDA_ON"] = '1'`
```sh
python webui.py
```
#### 祝你成功！

##### 你会发现在 `~/.local/lib/python3.10/site-packages/rwkv` 下面会出现一个 hip 目录，那是转换好的 CUDA 并行化算子

##### 失败了？全局搜索 `os.environ["RWKV_CUDA_ON"] = '1'` 改为 `os.environ["RWKV_CUDA_ON"] = '0'`

##### 改回来照样跑，又不是不能用，只不过少了个优雅的并行化，这锅我不背，ROCm生态是这样的，Pytorch能跑就已经不错了

### 📥 下载预训练权重
预训练权重存放于 `./weights/`
- RWKV-LM [RWKV-x060-World-1B6-v2-20240208-ctx4096.pth](https://huggingface.co/BlinkDL/rwkv-6-world/blob/main/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth)
- Visaul-RWKV-LM [rwkv1b5-vitl336p14-577token_mix665k_rwkv.pth](https://huggingface.co/howard-hou/visualrwkv-5/blob/main/rwkv1b5-vitl336p14-577token_mix665k_rwkv.pth)
- Visaul-RWKV [rwkv1b5-vitl336p14-577token_mix665k_visual.pth](https://huggingface.co/howard-hou/visualrwkv-5/blob/main/rwkv1b5-vitl336p14-577token_mix665k_visual.pth)
- Bert [s1bert.ckpt](https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s1bert25hz-2kh-longer-epoch%3D68e-step%3D50232.ckpt)
- HuBert [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)
- RMVPE [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt)

### 📝 修改预训练权重路径
- `./models/rwkv6/dialogue.py` 第 19 行
- `./models/rwkv6/continuation.py` 第 19 行
- `./models/music/run.py` 第 17 行
- `./models/language_test.py` 第 11 行
- `./models/visualRWKV/app/app_gpu.py`第 19 行 和 第 20 行
# 🧪 检验
- 执行
```sh 
python models/language_test.py
``` 
- 若可正常交互则说明准备工作无误



# 🚀 快速运行语言模型 （It is Available now!）(现在可以运行)
```sh
python webui.py
```
# 👀 快速运行RWKV视觉模型 （It is Available now!）(现在可以运行)
```sh
python webui.py
```
#### 调整语言模型运行策略在 `models/rwkv6/dialogue.py` 第 19 行，默认 "cuda fp16"
#### 调整视觉模型运行策略在 `models/visualRWKV/app/app_gpu.py` 第 24 行，默认 "cuda fp16"
Alic 这只北极熊比较菜，至少能跑

## 📂 项目结构

## 🧠 训练 (It is not Available now!)- 等等YuChuXi这只懒狐狸
- 训练需要 [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace/releases) 提取人脸特征，完成安装后在 `config/openseeface.json` 中配置路径即可
- 对于某些数据集可能需要自动语音标注 [DeepSpeech](https://github.com/mozilla/DeepSpeech)

### 📦 准备数据
你可以自己准备数据，也可以参考以下的数据集
- 语音、文字 [Mozilla Common Voice](https://commonvoice.mozilla.org/zh-CN)
- 语音、面部 [CelebV-Text](https://github.com/celebv-text/CelebV-Text)

### ⚙️ 数据预处理
- 视频或音频分片（默认 25FPS * 40s 一片，对应非语言模型的 25FPS * 1024CTX） `python`
- 提取 hubert 和 f0 `python`
- 提取视频的人脸特征 `python`

### 🎶 训练 T2F0
- 等等YuChuXi这只懒狐狸

### 🎶 训练 TF02M
- 等等YuChuXi这只懒狐狸

# 🌟 拓展
尝试 rwkv-music-demo
--
- 准备模型(选择 MIDI-model)
https://huggingface.co/BlinkDL/rwkv-5-music/tree/main
```sh
cd ./models/music
python ./run.py
```
- 模型路径在 `run.py` 的第 17 行，若无法正常运行修改第 22 行 "strategy='cuda fp32'" 为 "strategy='cpu fp32'"

State Tuning
--
参考 https://github.com/JL-er/RWKV-PEFT

rwkv-language-test
--
- 进入 `./models/rwkv/`
- 运行 `python language_test.py`

## ❓ 遇到问题?
- 如果无法启动webui，绝大多数可能是命令行终端无法连接Huggingface网站，请尝试使用代理，并在命令行终端中设置代理
```sh
export https_proxy=http://127.0.0.1:[port]
export http_proxy=http://127.0.0.1:[port]
```
- parselmouth 安装失败: 暂时将 `setuptools` 降级至 58.0 以下

## 其他

### 未来方向

### 致谢
- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- [Visual-RWKV](https://github.com/howard-hou/VisualRWKV)
- [SoftVC VITS 歌声转换](https://github.com/justinjohn0306/so-vits-svc-4.0/tree/4.0-v2)
- [GPT-SoVITS-WebUI](https://github.com/RVC-Boss/GPT-SoVITS)
- [RMVPE](https://github.com/Dream-High/RMVPE)
- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace)
- [DeepSpeech](https://github.com/mozilla/DeepSpeech)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/zh-CN)
- [CelebV-Text](https://github.com/celebv-text/CelebV-Text)
- [AMD](https://amd.com) 的计算卡