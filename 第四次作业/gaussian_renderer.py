import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H, self.W = image_height, image_width
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))

    def compute_projection(self, means3D, covs3d, K, R, t):
        N = means3D.shape[0]
        cam_points = means3D @ R.T + t.unsqueeze(0)
        depths = cam_points[:, 2].clamp(min=1.)
        screen_points = cam_points @ K.T
        means2D = screen_points[..., :2] / screen_points[..., 2:3]

        fx, fy = K[0, 0], K[1, 1]
        J_proj = torch.zeros((N, 2, 3), device=means3D.device)
        z = cam_points[:, 2].unsqueeze(1)
        x_coord = cam_points[:, 0].unsqueeze(1)
        y_coord = cam_points[:, 1].unsqueeze(1)

        J_proj[:, 0, 0] = fx / z.squeeze()
        J_proj[:, 0, 2] = -fx * x_coord.squeeze() / (z.squeeze() ** 2)
        J_proj[:, 1, 1] = fy / z.squeeze()
        J_proj[:, 1, 2] = -fy * y_coord.squeeze() / (z.squeeze() ** 2)

        covs_cam = torch.bmm(R.unsqueeze(0).expand(N, -1, -1),
                           torch.bmm(covs3d, R.T.unsqueeze(0).expand(N, -1, -1)))
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))
        return means2D, covs2D, depths

    def compute_gaussian_values(self, means2D, covs2D, pixels):
        """
        超省显存版本：逐像素、逐高斯点计算
        """
        N, H, W = means2D.shape[0], pixels.shape[0], pixels.shape[1]
        eps = 1e-6
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)

        det = torch.det(covs2D)
        inv = torch.inverse(covs2D)

        gaussian_values = torch.zeros((N, H, W), device=means2D.device)

        # 逐高斯点计算
        for i in range(N):
            dx = pixels - means2D[i]  # (H, W, 2)
            dx_flat = dx.reshape(-1, 2)  # (H*W, 2)
            
            # 计算 Σ^{-1} * (x-μ)
            temp = dx_flat @ inv[i]  # (H*W, 2)
            
            # 计算 (x-μ)^T * (Σ^{-1} * (x-μ))
            quad = (temp * dx_flat).sum(dim=1)  # (H*W,)
            
            # 计算高斯值
            norm = 1.0 / (2 * torch.pi * torch.sqrt(det[i]))
            g = norm * torch.exp(-0.5 * quad)
            gaussian_values[i] = g.reshape(H, W)

        return gaussian_values

    def forward(self, means3D, covs3d, colors, opacities, K, R, t):
        N = means3D.shape[0]
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)

        valid_mask = (depths > 1.) & (depths < 50.0)
        idx = torch.argsort(depths, descending=True)

        means2D = means2D[idx]
        covs2D = covs2D[idx]
        colors = colors[idx]
        opacities = opacities[idx]
        valid_mask = valid_mask[idx]

        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)

        alphas = opacities.view(N, 1, 1) * gaussian_values
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W).permute(0, 2, 3, 1)

        T = torch.ones((self.H, self.W), device=alphas.device)
        rendered = torch.zeros((self.H, self.W, 3), device=alphas.device)

        for i in range(N):
            weight = T * alphas[i]
            rendered = rendered + weight.unsqueeze(-1) * colors[i]
            T = T * (1.0 - alphas[i])

        return rendered