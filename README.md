# RRFVCRM  
一个以 RWKV 模型架构为基础的语音、动作生成AI

## 前言

### 初衷

### 约定
- 文档中的指令未经特别说明均位于项目根目录执行
- ```python``` 和 ```python3``` 是一样的

## 准备工作

### 检查环境

- 安装 [Python](https://python.org)
- 安装 CUDA/ROCm 和对应版本的 PyTorch
- 安装依赖库 ```pip install -r requirements.txt```

### 下载预训练权重
预训练权重存放于 ```weigths/pretrained/```
- RWKV-LM [RWKV-x060-World-1B6-v2-20240208-ctx4096.pth](https://huggingface.co/BlinkDL/rwkv-6-world/resolve/main/)
- Bert [s1bert25hz-2kh-longer-epoch%3D68e-step%3D50232.ckpt](https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/)
- RMVPE [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/)

## 快速开始
- 下载示例模型 [啥都没有](https://nothing)
- 下载训练有素的VC模型 [啥都没有](https://nothing)
- 启动！ ```python webui.py```

## 项目结构

## 训练

### 准备数据

### 数据预处理

### 训练 T2F0

### 训练 TF02M

### 拓展

## 遇到问题?

## 其他

### 未来方向

### 致谢