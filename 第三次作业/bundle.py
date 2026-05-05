import numpy as np
import torch
import matplotlib.pyplot as plt

# =========================
# Euler -> Rotation Matrix
# =========================
def euler_to_matrix(euler):
    rx, ry, rz = euler[:, 0], euler[:, 1], euler[:, 2]

    cosx, sinx = torch.cos(rx), torch.sin(rx)
    cosy, siny = torch.cos(ry), torch.sin(ry)
    cosz, sinz = torch.cos(rz), torch.sin(rz)

    Rx = torch.stack([
        torch.stack([torch.ones_like(rx), torch.zeros_like(rx), torch.zeros_like(rx)], dim=-1),
        torch.stack([torch.zeros_like(rx), cosx, -sinx], dim=-1),
        torch.stack([torch.zeros_like(rx), sinx, cosx], dim=-1),
    ], dim=-2)

    Ry = torch.stack([
        torch.stack([cosy, torch.zeros_like(ry), siny], dim=-1),
        torch.stack([torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry)], dim=-1),
        torch.stack([-siny, torch.zeros_like(ry), cosy], dim=-1),
    ], dim=-2)

    Rz = torch.stack([
        torch.stack([cosz, -sinz, torch.zeros_like(rz)], dim=-1),
        torch.stack([sinz, cosz, torch.zeros_like(rz)], dim=-1),
        torch.stack([torch.zeros_like(rz), torch.zeros_like(rz), torch.ones_like(rz)], dim=-1),
    ], dim=-2)

    return Rz @ Ry @ Rx


# =========================
# 读取数据
# =========================
data = np.load("points2d.npz")
print("Keys:", data.files)

view_keys = sorted(data.files)  # ['view_000', ..., 'view_049']

points2d_list = []
visibility_list = []

for k in view_keys:
    arr = data[k]  # 可能是 (N,2) 或 (N,3)

    if arr.shape[1] == 2:
        pts = arr
        vis = np.ones(arr.shape[0])
    elif arr.shape[1] == 3:
        pts = arr[:, :2]
        vis = arr[:, 2]
    else:
        raise ValueError(f"Unexpected shape in {k}: {arr.shape}")

    points2d_list.append(pts)
    visibility_list.append(vis)

points2d = torch.tensor(np.stack(points2d_list), dtype=torch.float32)
visibility = torch.tensor(np.stack(visibility_list), dtype=torch.float32)

visibility = data.get("visibility", None)
if visibility is None:
    visibility = torch.ones(points2d.shape[:2])
else:
    visibility = torch.tensor(visibility, dtype=torch.float32)

colors = np.load("points3d_colors.npy")

# =========================
# 参数
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

points2d = points2d.to(device)
visibility = visibility.to(device)

B, N, _ = points2d.shape

cx, cy = 256, 256  # 如果不是512图像请改

# =========================
# 初始化
# =========================
points_3d = torch.randn(N, 3, device=device) * 0.1
points_3d.requires_grad_(True)

euler_angles = torch.zeros(B, 3, device=device, requires_grad=True)

translations = torch.zeros(B, 3, device=device)
translations[:, 2] = -2.5
translations.requires_grad_(True)

log_f = torch.tensor(6.0, device=device, requires_grad=True)

# =========================
# 投影
# =========================
def project(points_3d, R, T, f, cx, cy):
    points_cam = torch.matmul(R, points_3d.T).transpose(1, 2) + T[:, None, :]

    Xc = points_cam[..., 0]
    Yc = points_cam[..., 1]
    Zc = points_cam[..., 2]

    Zc = Zc.clamp(max=-1e-3)

    u = -f * (Xc / Zc) + cx
    v =  f * (Yc / Zc) + cy

    return torch.stack([u, v], dim=-1)

# =========================
# Huber Loss
# =========================
def huber(x, delta=1.0):
    abs_x = torch.abs(x)
    return torch.where(abs_x < delta, 0.5 * x**2, delta * (abs_x - 0.5 * delta))

# =========================
# 优化
# =========================
optimizer = torch.optim.Adam([
    points_3d,
    euler_angles,
    translations,
    log_f
], lr=1e-3)

loss_history = []

for i in range(1000):
    optimizer.zero_grad()

    R = euler_to_matrix(euler_angles)
    f = torch.exp(log_f)

    pred = project(points_3d, R, translations, f, cx, cy)

    diff = (pred - points2d) * visibility[..., None]
    loss = huber(diff).sum() / (visibility.sum() + 1e-6)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if i % 50 == 0:
        print(f"{i}: loss={loss.item():.4f}, f={f.item():.2f}")

# =========================
# 可视化
# =========================
plt.plot(loss_history)
plt.title("Loss Curve")
plt.show()

# =========================
# 保存 OBJ
# =========================
def save_obj(path, pts, cols):
    with open(path, "w") as f:
        for p, c in zip(pts, cols):
            f.write(f"v {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

pts = points_3d.detach().cpu().numpy()
save_obj("result.obj", pts, colors)

print("✅ Done: result.obj")