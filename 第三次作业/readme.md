# 3D Head Reconstruction via Bundle Adjustment

This repository is the official implementation of **3D Head Reconstruction via Bundle Adjustment**.

📌 本项目从多视角 2D 观测出发，通过优化恢复：

* 3D 点云结构（20000 points）
* 相机外参（50 views）
* 相机内参（共享焦距）

<p align="center">
  <img src="docs/pipeline.png" width="600"/>
</p>

---

# Requirements

To install requirements:

```bash
pip install -r requirements.txt
📋 环境配置说明：
推荐使用 Conda：
bash
运行
conda create -n ba python=3.9
conda activate ba
pip install torch numpy matplotlib tqdm
📦 数据准备：
将以下文件放在项目根目录：
plaintext
points2d.npz
points3d_colors.npy
Training (Optimization)
To run Bundle Adjustment optimization:
bash
运行
python bundle.py
📋 训练（优化）说明：
本项目不涉及传统深度学习训练，而是基于 可微优化（Bundle Adjustment）：
优化变量：
3D 点：(20000, 3)
相机旋转（Euler）：(50, 3)
相机平移：(50, 3)
焦距：f
投影模型：
u=−f⋅ 
Z 
c
​
 
X 
c
​
 
​
 +c 
x
​
 ,v=f⋅ 
Z 
c
​
 
Y 
c
​
 
​
 +c 
y
​
 
其中：
[X 
c
​
 ,Y 
c
​
 ,Z 
c
​
 ]=R⋅X+T
损失函数：
重投影误差（Huber Loss）
可见性 mask 加权
超参数：
bash
运行
iterations = 800
lr = 1e-3
init focal ≈ 700
init translation = [0, 0, -2.5]
Evaluation
To evaluate reconstruction quality:
bash
运行
python bundle.py
📋 评估方式：
由于无 GT 3D，本项目采用：
📉 重投影误差（Reprojection Loss）
👁️ 可视化结果（OBJ 点云）
输出文件：
plaintext
reconstructed_head.obj
Pre-trained Models
本项目为优化问题，不涉及预训练模型。
但可以直接使用：
输入：points2d.npz
输出：优化后的 3D 结构
Results
📊 Optimization Performance
表格
Iteration	Loss
0	238034
100	11201
300	1147
500	923
800	566
最终焦距：
plaintext
f ≈ 700
<p align="center">
<img src="loss_curve.png" width="600" alt="Loss Curve">
</p>
🧠 Reconstruction Result
生成点云：
plaintext
reconstructed_head.obj
可用以下软件查看：
MeshLab（推荐）
Blender
CloudCompare
🎬 Demo Video
<p align="center">
<video src="bandicam 2026-05-06 04-00-24-957.mp4" width="800" controls>
Your browser does not support the video tag.
</video>
</p>
Visualization
Loss 曲线：
bash
运行
python bundle.py
将自动绘制优化过程：
📉 Loss 持续下降，说明 BA 收敛良好
Key Features
✅ 无需 PyTorch3D
✅ 支持 Euler 角旋转
✅ 数值稳定（Z clamp + log focal）
✅ 支持 visibility mask
✅ 输出彩色点云
Project Structure
plaintext
.
├── bundle.py
├── points2d.npz
├── points3d_colors.npy
├── loss_curve.png
├── bandicam 2026-05-06 04-00-24-957.mp4
├── reconstructed_head.obj
└── README.md
plaintext

---

### ✅ 改动说明
1.  在 `Results` 章节，直接插入了 `loss_curve.png` 的图片引用，和你项目里的文件对应
2.  新增了 `Demo Video` 部分，用 HTML 视频标签嵌入了你的录屏文件 `bandicam 2026-05-06 04-00-24-957.mp4`
3.  更新了 `Project Structure` 部分，把新增的文件也列了进去

你只需要确保 `loss_curve.png` 和视频文件和 `README.md` 放在同一目录下，就能正常显示啦。

需要我帮你再把项目的「使用说明」和「效果展示」部分润色得更专业、更适合放到 GitHub 上吗？
写一份完整的markdown格式readme
下面是一份完整、专业、可直接用于 GitHub 的 Markdown 版 README，包含了你的项目所有信息，也嵌入了你的 loss 曲线和演示视频：
markdown
# 🎯 3D Head Reconstruction via Bundle Adjustment

从多视角 2D 观测数据出发，通过可微优化实现人脸三维重建，同时恢复相机内参、外参和三维点云结构。

---

## 📖 项目简介

本项目基于 Bundle Adjustment（光束平差法），对人脸多视角 2D 关键点进行优化，同时求解：

- ✅ 三维点云结构（约 20000 个点）
- ✅ 相机外参（50 个视角的旋转与平移）
- ✅ 相机内参（共享焦距）

不依赖 PyTorch3D，仅用纯 PyTorch 实现，优化过程透明、可复现。

<p align="center">
  <img src="docs/pipeline.png" width="600" alt="pipeline">
</p>

---

## 🧩 环境配置

推荐使用 Conda 搭建环境：

```bash
# 创建并激活环境
conda create -n ba python=3.9 -y
conda activate ba

# 安装依赖
pip install torch numpy matplotlib tqdm
📂 数据准备
将以下文件放在项目根目录：
text
./
├── points2d.npz           # 2D 关键点观测 + 可见性
├── points3d_colors.npy    # 每个 3D 点对应的 RGB 颜色
└── bundle.py              # 主优化脚本
points2d.npz：每个视角的关键点 2D 坐标与可见性 mask
points3d_colors.npy：与 3D 点一一对应的颜色信息，用于生成彩色点云
🚀 快速开始
运行优化（Bundle Adjustment）：
bash
运行
python bundle.py
运行后会自动生成：
loss_curve.png：优化过程的重投影误差曲线
reconstructed_head.obj：带颜色的三维点云模型
⚙️ 优化原理
1. 优化变量
表格
变量	维度	说明
3D 点云	(20000, 3)	待重建的三维结构
相机旋转	(50, 3)	每个视角的欧拉角（XYZ）
相机平移	(50, 3)	每个视角的平移向量
焦距 f	标量	所有相机共享的内参
2. 投影模型
相机坐标系变换：
[X 
c
​
 ,Y 
c
​
 ,Z 
c
​
 ] 
T
 =R⋅X+T
针孔相机投影：
u=−f⋅ 
Z 
c
​
 
X 
c
​
 
​
 +c 
x
​
 ,v=f⋅ 
Z 
c
​
 
Y 
c
​
 
​
 +c 
y
​
 
其中 
c 
x
​
 ,c 
y
​
 
 为图像主点（(W/2, H/2)）。
3. 损失函数
采用重投影误差作为目标函数，并用可见性 mask 过滤无效点：
L= 
N
1
​
 ∑ 
i=1
N
​
 ∑ 
j=1
V
​
 vis 
ij
​
 ⋅∥(u 
ij
​
 ,v 
ij
​
 )−( 
u
^
  
ij
​
 , 
v
^
  
ij
​
 )∥ 
2
2
​
 
u 
ij
​
 ,v 
ij
​
 
：观测到的 2D 关键点
u
^
  
ij
​
 , 
v
^
  
ij
​
 
：由当前 3D 点和相机参数投影得到的预测坐标
vis 
ij
​
 
：可见性 mask（0 表示该点在当前视角不可见）
4. 关键超参数
python
运行
NUM_ITERATIONS = 800
LEARNING_RATE = 1e-3
INIT_FOCAL = 700.0
INIT_TRANSLATION_Z = -2.5
📊 优化过程与结果
1. 重投影误差变化
表格
迭代次数	损失值
0	238034
100	11201
300	1147
500	923
800	566
<p align="center">
<img src="loss_curve.png" width="600" alt="Loss Curve">
</p>
损失曲线持续下降，说明优化收敛稳定
最终焦距收敛到约 700，与初始估计一致
2. 三维重建结果
运行完成后会生成 reconstructed_head.obj，这是一个带颜色的点云模型，可在以下软件中查看：
Blender：导入 .obj，开启「顶点颜色」显示
MeshLab：直接打开即可查看彩色点云
CloudCompare：适合专业点云分析
🎬 效果演示
<p align="center">
<video src="bandicam 2026-05-06 04-00-24-957.mp4" width="800" controls>
您的浏览器不支持视频播放，请点击下载查看演示视频。
</video>
</p>
