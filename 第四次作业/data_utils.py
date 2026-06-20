import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import struct
from plyfile import PlyData

def sample_farthest_points(points: torch.Tensor, K: int) -> torch.Tensor:
    """
    随机选择 K 个点（简化版，避免复杂计算）
    """
    device = points.device
    N = points.shape[0]
    
    if K >= N:
        return torch.arange(N, device=device)
    
    # 随机选择 K 个点
    indices = torch.randperm(N, device=device)[:K]
    return indices

class ColmapDataset(torch.utils.data.Dataset):
    def __init__(self, colmap_dir: str):
        super().__init__()
        self.colmap_dir = Path(colmap_dir)
        self.sparse_dir = self.colmap_dir / "sparse" / "0"
        self.images_dir = self.colmap_dir / "images"
        
        self._load_camera_info()
        self._load_images_info()
        self._load_point_cloud()
        print(f"Loaded {len(self.image_files)} images")
        print(f"Loaded {len(self.points3D_xyz)} 3D points")

    def _load_camera_info(self):
        cameras_file = self.sparse_dir / "cameras.txt"
        if not cameras_file.exists():
            cameras_file = self.sparse_dir / "cameras.bin"
            if cameras_file.exists():
                self._load_camera_bin(cameras_file)
                return
            raise FileNotFoundError("cameras.txt or cameras.bin not found")
        
        with open(cameras_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
                    self.width = int(parts[2])
                    self.height = int(parts[3])
                    fx, fy, cx, cy = map(float, parts[4:8])
                    self.K = torch.tensor([[fx, 0, cx],
                                          [0, fy, cy],
                                          [0, 0, 1]], dtype=torch.float32)
                    break

    def _load_camera_bin(self, path):
        with open(path, 'rb') as f:
            # 跳过头部
            struct.unpack('<Q', f.read(8))[0]  # num_cameras
            camera_id = struct.unpack('<i', f.read(4))[0]
            model_id = struct.unpack('<i', f.read(4))[0]
            width = struct.unpack('<Q', f.read(8))[0]
            height = struct.unpack('<Q', f.read(8))[0]
            params = []
            for _ in range(4):
                params.append(struct.unpack('<d', f.read(8))[0])
        
        self.width, self.height = width, height
        self.K = torch.tensor([[params[0], 0, params[2]],
                              [0, params[1], params[3]],
                              [0, 0, 1]], dtype=torch.float32)

    def _load_images_info(self):
        images_file = self.sparse_dir / "images.txt"
        self.image_files = []
        self.R_matrices = []
        self.t_vectors = []
        
        with open(images_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#'):
                i += 1
                continue
            
            parts = line.split()
            if len(parts) >= 10:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                R = self._quaternion_to_matrix(qw, qx, qy, qz)
                self.R_matrices.append(R)
                self.t_vectors.append([tx, ty, tz])
                self.image_files.append(parts[9])
            
            i += 2

    def _load_point_cloud(self):
        # 优先尝试 PLY 文件
        ply_file = self.sparse_dir / "points3D.ply"
        if ply_file.exists():
            self._load_points_from_ply(ply_file)
            return
        
        # 其次尝试 TXT 文件
        txt_file = self.sparse_dir / "points3D.txt"
        if txt_file.exists():
            self._load_points_from_txt(txt_file)
            return
            
        raise FileNotFoundError("Neither points3D.ply nor points3D.txt found")

    def _load_points_from_ply(self, ply_file):
        plydata = PlyData.read(ply_file)
        vertex = plydata['vertex']
        self.points3D_xyz = torch.tensor(
            np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1),
            dtype=torch.float32
        )
        if 'red' in vertex.data.dtype.names:
            self.points3D_rgb = torch.tensor(
                np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1),
                dtype=torch.float32
            )
        else:
            self.points3D_rgb = torch.ones_like(self.points3D_xyz) * 255

    def _load_points_from_txt(self, txt_file):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        points_xyz = []
        points_rgb = []
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                x, y, z = map(float, parts[1:4])
                r, g, b = map(float, parts[4:7])
                points_xyz.append([x, y, z])
                points_rgb.append([r, g, b])
        self.points3D_xyz = torch.tensor(points_xyz, dtype=torch.float32)
        self.points3D_rgb = torch.tensor(points_rgb, dtype=torch.float32)

    def _quaternion_to_matrix(self, qw, qx, qy, qz):
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])
        return torch.tensor(R, dtype=torch.float32)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.images_dir / self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ================== 关键修复：强制降低分辨率 ==================
        target_h, target_w = 225, 300  # 16:9 比例，非常小
        image = cv2.resize(image, (target_w, target_h))
        
        # 调整相机内参
        scale_x = target_w / self.width
        scale_y = target_h / self.height
        K = self.K.clone()
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] *= scale_x  # cx
        K[1, 2] *= scale_y  # cy
        # ============================================================
        
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image, dtype=torch.float32)
        
        R = self.R_matrices[idx]
        t = torch.tensor(self.t_vectors[idx], dtype=torch.float32)
        
        return {
            'image': image,
            'K': K,
            'R': R,
            't': t,
            'image_path': str(img_path)
        }