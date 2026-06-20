import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import cv2
import os

from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer
from data_utils import ColmapDataset, sample_farthest_points

@dataclass
class TrainConfig:
    num_epochs: int = 200
    batch_size: int = 1
    learning_rate: float = 0.001
    grad_clip: float = 1.0
    save_every: int = 20
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    debug_every: int = 1
    debug_samples: int = 4

class GaussianTrainer:
    def __init__(self, model, renderer, config, device):
        self.model = model.to(device)
        self.renderer = renderer.to(device)
        self.config = config
        self.device = device
        
        optable_params = [
            {'params': [self.model.positions], 'lr': 0.000016, "name": "xyz"},
            {'params': [self.model.colors], 'lr': 0.025, "name": "color"},
            {'params': [self.model.opacities], 'lr': 0.05, "name": "opacity"},
            {'params': [self.model.scales], 'lr': 0.005, "name": "scaling"},
            {'params': [self.model.rotations], 'lr': 0.001, "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(optable_params, lr=0.001, eps=1e-15)
        
        Path(config.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        Path(config.log_dir).mkdir(exist_ok=True, parents=True)
        self.debug_indices = None

    def save_debug_images(self, epoch, rendered_images, gt_images, image_paths):
        rendered = rendered_images.detach().cpu().numpy()
        gt = gt_images.detach().cpu().numpy()
        gt_cells, rendered_cells = [], []
        for b in range(rendered.shape[0]):
            r = (rendered[b] * 255).clip(0, 255).astype(np.uint8)
            g = (gt[b] * 255).clip(0, 255).astype(np.uint8)
            r = cv2.cvtColor(r, cv2.COLOR_RGB2BGR)
            g = cv2.cvtColor(g, cv2.COLOR_RGB2BGR)
            label = Path(image_paths[b]).stem
            cv2.putText(g, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            gt_cells.append(g)
            rendered_cells.append(r)
        gt_row = np.concatenate(gt_cells, axis=1)
        rendered_row = np.concatenate(rendered_cells, axis=1)
        cv2.putText(gt_row, "GT", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(rendered_row, "Rendered", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        grid = np.concatenate([gt_row, rendered_row], axis=0)
        output_path = Path(self.config.log_dir) / f"epoch_{epoch:04d}.png"
        cv2.imwrite(str(output_path), grid)

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        path = Path(self.config.checkpoint_dir) / f"checkpoint_{epoch:06d}.pt"
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def visualize_rendering(self, dataset, save_vid_path, num_frames=300):
        print("Generating rendering visualization...")
        sample = dataset[0]
        K = sample['K'].to(self.device)
        H, W = sample['image'].shape[:2]
        out = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 3, (W*2, H))
        with torch.no_grad():
            gaussian_params = self.model()
        for data_item in tqdm(dataset, desc="Rendering frames"):
            R_torch = data_item['R'].to(self.device)
            t_torch = data_item['t'].to(self.device).reshape(-1, 3)
            with torch.no_grad():
                rendered_image = self.renderer(
                    means3D=gaussian_params['positions'],
                    covs3d=gaussian_params['covariance'],
                    colors=gaussian_params['colors'],
                    opacities=gaussian_params['opacities'],
                    K=K.squeeze(0),
                    R=R_torch.squeeze(0),
                    t=t_torch.squeeze(0),
                )
            frame = rendered_image.cpu().numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            ori_img = (data_item['image']*255).cpu().numpy().astype(np.uint8)
            vis = cv2.cvtColor(np.concatenate((ori_img, frame), axis=1), cv2.COLOR_RGB2BGR)
            out.write(vis)
        out.release()
        print(f"Video saved to: {save_vid_path}")

    def train_step(self, batch, in_train=True):
        images = batch['image'].to(self.device)
        K = batch['K'].to(self.device)
        R = batch['R'].to(self.device)
        t = batch['t'].to(self.device).reshape(-1, 3)
        gaussian_params = self.model()
        rendered_images = self.renderer(
            means3D=gaussian_params['positions'],
            covs3d=gaussian_params['covariance'],
            colors=gaussian_params['colors'],
            opacities=gaussian_params['opacities'],
            K=K.squeeze(0),
            R=R.squeeze(0),
            t=t.squeeze(0),
        ).unsqueeze(0)
        if not in_train:
            return rendered_images
        loss = torch.abs(rendered_images - images).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return loss.item(), rendered_images

    def train(self, train_loader):
        if self.debug_indices is None:
            dataset_size = len(train_loader.dataset)
            self.debug_indices = np.random.choice(
                dataset_size,
                min(self.config.debug_samples, dataset_size),
                replace=False
            )
        for epoch in range(self.config.num_epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            epoch_loss, num_batches = 0.0, 0
            for batch in pbar:
                loss, rendered_images = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                pbar.set_postfix({'loss': f"{epoch_loss/num_batches:.4f}"})
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch)
            if epoch % self.config.debug_every == 0:
                rendered_list, gt_list, path_list = [], [], []
                for idx in self.debug_indices:
                    sample = train_loader.dataset[idx]
                    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in sample.items()}
                    with torch.no_grad():
                        rendered = self.train_step(batch, in_train=False)
                    rendered_list.append(rendered.squeeze(0))
                    gt_list.append(sample['image'])
                    path_list.append(sample['image_path'])
                self.save_debug_images(
                    epoch=epoch,
                    rendered_images=torch.stack(rendered_list, dim=0),
                    gt_images=torch.stack(gt_list, dim=0),
                    image_paths=path_list
                )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--debug_every', type=int, default=1)
    parser.add_argument('--debug_samples', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    config = TrainConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=os.path.join(args.checkpoint_dir, "debug_images"),
        debug_every=args.debug_every,
        debug_samples=args.debug_samples
    )
    dataset = ColmapDataset(args.colmap_dir)
    
    # ================== 关键修复：点云下采样到 1000 ==================
    num_points = 1000
    if len(dataset.points3D_xyz) > num_points:
        indices = sample_farthest_points(dataset.points3D_xyz, num_points)
        dataset.points3D_xyz = dataset.points3D_xyz[indices]
        dataset.points3D_rgb = dataset.points3D_rgb[indices]
        print(f"Downsampled points to {num_points}")
    # ================================================================
    
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    sample = dataset[0]['image']
    H, W = sample.shape[:2]
    model = GaussianModel(dataset.points3D_xyz, dataset.points3D_rgb)
    renderer = GaussianRenderer(image_height=H, image_width=W)
    trainer = GaussianTrainer(model, renderer, config, device)
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        config.num_epochs -= start_epoch
    print("Starting training...")
    trainer.train(train_loader)
    print("Training completed!")
    trainer.visualize_rendering(dataset, os.path.join(args.checkpoint_dir, "debug_rendering.mp4"))

if __name__ == "__main__":
    main()