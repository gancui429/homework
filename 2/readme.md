# 泊松图像融合作业报告

## 一、项目概述
本项目基于 PyTorch 实现泊松图像融合（Poisson Blending），并使用 Gradio 搭建交互式可视化界面。
支持手动绘制多边形选区、预览偏移位置、自动将前景图像无缝融合到背景图像中。

---

## 二、环境搭建步骤

### 1. 创建虚拟环境（推荐，可选）
使用 Conda 创建独立环境，避免依赖冲突：
```bash
conda create -n poisson_env python=3.9
conda activate poisson_env
pip install torch numpy pillow gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import torch
import numpy
import PIL
import gradio

print("PyTorch 版本：", torch.__version__)
print("NumPy 版本：", numpy.__version__)
print("Pillow 版本：", PIL.__version__)
print("Gradio 版本：", gradio.__version__)
print("CUDA 是否可用：", torch.cuda.is_available())
