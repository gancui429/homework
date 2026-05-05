# 📄 我的论文标题  
## 基于 COLMAP 的 50 张渲染图三维重建实现

---

# 📌 项目简介

本项目使用 COLMAP 对 50 张渲染图像进行三维重建，完整流程如下：

- 特征提取（SIFT）
- 特征匹配（Exhaustive Matching）
- 稀疏重建（SfM + Bundle Adjustment）
- 稠密重建（MVS）
- 点云输出（.ply）

---

# 📊 流程图

输入图像（50张渲染图）  
↓  
特征提取（SIFT）  
↓  
特征匹配（Exhaustive Matcher）  
↓  
稀疏重建（SfM + BA）  
↓  
稠密重建（MVS）  
↓  
输出点云（.ply）

---

# 📦 环境配置

## 1️⃣ COLMAP 安装（Windows）

下载 COLMAP：

https://github.com/colmap/colmap

解压路径示例：

D:\colmap\

加入系统环境变量 PATH：

D:\colmap\

验证安装：

colmap -h

---

## 2️⃣ Python依赖（可选）

pip install numpy opencv-python open3d matplotlib

---

# 🗂 数据准备

数据结构如下：

data/
├── images/        50张输入图像
├── database.db
├── sparse/
├── dense/

---

# 🚀 重建流程

## 1️⃣ 特征提取

colmap feature_extractor ^
  --database_path data/database.db ^
  --image_path data/images ^
  --ImageReader.single_camera 1 ^
  --SiftExtraction.use_gpu 1

---

## 2️⃣ 特征匹配

colmap exhaustive_matcher ^
  --database_path data/database.db ^
  --SiftMatching.use_gpu 1

---

## 3️⃣ 稀疏重建

colmap mapper ^
  --database_path data/database.db ^
  --image_path data/images ^
  --output_path data/sparse

---

## 4️⃣ 稠密重建

### 图像去畸变

colmap image_undistorter ^
  --image_path data/images ^
  --input_path data/sparse/0 ^
  --output_path data/dense ^
  --output_type COLMAP

### PatchMatch Stereo

colmap patch_match_stereo ^
  --workspace_path data/dense ^
  --workspace_format COLMAP ^
  --PatchMatchStereo.use_gpu 1

### 点云融合

colmap stereo_fusion ^
  --workspace_path data/dense ^
  --workspace_format COLMAP ^
  --input_type geometric ^
  --output_path data/dense/fused.ply

---

# 📊 结果

稀疏点云：

![稀疏]

稠密点云：

data/dense/fused.ply

---


