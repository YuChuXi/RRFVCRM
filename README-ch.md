#
![RRFVCM](assets/logo.jpg)



## 前言
一个以 RWKV 模型架构为基础的语音、动作生成AI
### 初衷

### 约定
- 文档中的指令未经特别说明均位于项目根目录执行
- ```python``` 和 ```python3``` 是一样的

## 准备工作

### 配置环境
- 安装 [Python](https://python.org)
- 安装 CUDA/ROCm 和对应版本的 PyTorch
- 安装依赖库 ```pip install -r requirements.txt```

### 下载预训练权重
预训练权重存放于 ```weigths/pretrained/```
rwkv1b6语言模型(RWKV-LM)放在```./models/rwkv/rwkv-lm-models```
- RWKV-LM [RWKV-x060-World-1B6-v2-20240208-ctx4096.pth](https://huggingface.co/BlinkDL/rwkv-6-world/blob/main/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth)
- Bert [s1bert.ckpt](https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s1bert25hz-2kh-longer-epoch%3D68e-step%3D50232.ckpt)
- HuBert [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)
- RMVPE [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt)

### 检验
- 执行 ```python  models/language_test.py``` 若可正常交互则说明准备工作无误

## 快速开始
- 下载示例模型 [啥都没有](https://nothing)
- 下载训练有素的VC模型 [啥都没有](https://nothing)
- 启动！ ```python webui.py```

## 项目结构

## 训练
- 训练需要 [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace/releases) 提取人脸特征，完成安装后在 ```config/openseeface.json``` 中配置路径即可
- 对于某些数据集可能需要自动语音标注 [DeepSpeech]](https://github.com/mozilla/DeepSpeech)

### 准备数据
你可以自己准备数据，也可以参考以下的数据集
- 语音、文字 [Mozilla Common Voice](https://commonvoice.mozilla.org/zh-CN)
- 语音、面部 [CelebV-Text](https://github.com/celebv-text/CelebV-Text)

### 数据预处理
- 视频或音频分片（默认25FPS * 40s 一片，对应非语言模型的 25FPS * 1024CTX） ```python ```
- 提取 hubert 和 f0 ```python ```
- 提取视频的人脸特征 ```python ```

### 训练 T2F0

### 训练 TF02M

# 拓展
rwkv-music-demo
--
- 准备模型(选择MIDI-model)
https://huggingface.co/BlinkDL/rwkv-5-music/tree/main
- 进入```./music```
- 运行 ```python ./run.py```
- 模型路径在```run.py```的第17行，若无法正常运行修改第22行"strategy='cuda fp32'"为"strategy='cpu fp32'"

rwkv-language-test
--
- 进入 ```./models/rwkv/```
- 运行 ```python language_test.py```
## 遇到问题?
- parselmouth 安装失败: 暂时将 ```setuptools``` 降级至 58.0 以下

## 其他

### 未来方向

### 致谢
- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- [SoftVC VITS 歌声转换](https://github.com/justinjohn0306/so-vits-svc-4.0/tree/4.0-v2)
- [GPT-SoVITS-WebUI](https://github.com/RVC-Boss/GPT-SoVITS)
- [RMVPE](https://github.com/Dream-High/RMVPE)
- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace)
- [DeepSpeech](https://github.com/mozilla/DeepSpeech)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/zh-CN)
- [CelebV-Text](https://github.com/celebv-text/CelebV-Text)
- [AMD](https://amd.com) 的计算卡
