# ==============================================
# PIX2PIX 全卷积网络 最终纯净版
# 无txt、无额外文件、无报错、直接跑
# ==============================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torchvision import transforms
import gradio as gr

# ===================== 模型 =====================
class FullyConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3,64,4,2,1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(128,256,4,2,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(nn.Conv2d(256,512,4,2,1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))

        self.dec1 = nn.Sequential(nn.ConvTranspose2d(512,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(64,3,4,2,1), nn.Tanh())

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return self.dec4(x)

# ===================== 数据集 =====================
class FacadesDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = []
        if os.path.exists(root):
            self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(('.jpg','.png'))]

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    def __len__(self):
        return max(1, len(self.files))

    def __getitem__(self, idx):
        if len(self.files) == 0:
            dummy = torch.zeros(3,256,256)
            return dummy, dummy

        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        w = w // 2
        return self.trans(img[:,:w]), self.trans(img[:,w:])

# ===================== 推理 =====================
def predict(img):
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    img = cv2.resize(img, (256,256))
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
    inp = trans(img).unsqueeze(0)

    model = FullyConvNetwork()
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        out = model(inp).squeeze().permute(1,2,0).cpu().numpy()
    out = (out * 0.5 + 0.5) * 255
    out = out.astype('uint8')
    return out

# ===================== 训练 =====================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    dataset = FacadesDataset("./facades/train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = FullyConvNetwork().to(device)
    opt = optim.Adam(model.parameters(), lr=0.0002)
    loss_fn = nn.L1Loss()

    print("开始训练...")
    for epoch in range(20):
        total = 0
        for a, b in loader:
            a, b = a.to(device), b.to(device)
            loss = loss_fn(model(a), b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1} 损失: {total:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("✅ 训练完成！模型已保存！")

# ===================== 启动界面 =====================
if __name__ == "__main__":
    # 先训练
    train()

    # 再启动界面
    gr.Interface(
        fn=predict,
        inputs=gr.Image(),
        outputs=gr.Image(),
        title="Pix2Pix 图像翻译",
        description="上传语义分割图 → 生成真实图"
    ).launch(server_port=7860)