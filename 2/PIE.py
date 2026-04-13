import gradio as gr
from PIL import ImageDraw, Image
import numpy as np
import torch

# 初始化多边形
def initialize_polygon():
    return {'points': [], 'closed': False}

# 添加点
def add_point(img_original, polygon_state, evt: gr.SelectData):
    if polygon_state['closed']:
        return img_original, polygon_state
    x, y = evt.index
    polygon_state['points'].append((x, y))
    img_with_poly = img_original.copy()
    draw = ImageDraw.Draw(img_with_poly)
    if len(polygon_state['points']) > 1:
        draw.line(polygon_state['points'], fill='red', width=2)
    for (px, py) in polygon_state['points']:
        draw.ellipse((px-3, py-3, px+3, py+3), fill='blue')
    return img_with_poly, polygon_state

# 闭合多边形
def close_polygon(img_original, polygon_state):
    if not polygon_state['closed'] and len(polygon_state['points']) > 2:
        polygon_state['closed'] = True
        img_with_poly = img_original.copy()
        draw = ImageDraw.Draw(img_with_poly)
        draw.polygon(polygon_state['points'], outline='red')
        return img_with_poly, polygon_state
    return img_original, polygon_state

# 更新背景预览
def update_background(background_image_original, polygon_state, dx, dy):
    if background_image_original is None or not polygon_state['closed']:
        return background_image_original
    img = background_image_original.copy()
    draw = ImageDraw.Draw(img)
    shifted = [(x+dx, y+dy) for x,y in polygon_state['points']]
    draw.polygon(shifted, outline='red')
    return img

# 生成mask
def create_mask_from_points(points, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 3:
        return mask
    pil_mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(pil_mask)
    draw.polygon([(p[0], p[1]) for p in points], fill=255)
    return np.array(pil_mask)

# 拉普拉斯损失
def cal_laplacian_loss(fg_img, fg_mask, blend_img, bg_mask):
    lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=fg_img.device)
    lap = lap.unsqueeze(0).unsqueeze(0).expand(3,1,3,3)
    fg_lap = torch.nn.functional.conv2d(fg_img, lap, padding=1, groups=3)
    bg_lap = torch.nn.functional.conv2d(blend_img, lap, padding=1, groups=3)
    diff = (fg_lap * fg_mask - bg_lap * bg_mask) ** 2
    loss = diff.sum()
    n = bg_mask.sum()
    return loss / n if n > 0 else loss

# 泊松融合（纯PyTorch，无scipy）
def blending(fg_img, bg_img, dx, dy, poly_state):
    if not poly_state['closed'] or fg_img is None or bg_img is None:
        return bg_img
    fg = np.array(fg_img)
    bg = np.array(bg_img)
    pts = np.array(poly_state['points'])
    bg_pts = pts + [int(dx), int(dy)]
    fg_mask = create_mask_from_points(pts, fg.shape[0], fg.shape[1])
    bg_mask = create_mask_from_points(bg_pts, bg.shape[0], bg.shape[1])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fg_t = torch.from_numpy(fg).to(device).permute(2,0,1).unsqueeze(0).float()/255
    bg_t = torch.from_numpy(bg).to(device).permute(2,0,1).unsqueeze(0).float()/255
    fg_m_t = torch.from_numpy(fg_mask).to(device).unsqueeze(0).unsqueeze(0).float()/255
    bg_m_t = torch.from_numpy(bg_mask).to(device).unsqueeze(0).unsqueeze(0).float()/255
    out = bg_t.clone()
    m = bg_m_t.bool().expand(-1,3,-1,-1)
    out[m] = out[m] * 0.9 + fg_t[fg_m_t.bool().expand(-1,3,-1,-1)] * 0.1
    out.requires_grad = True
    opt = torch.optim.Adam([out], lr=1e-2)
    steps = 1500  # 小步数，CPU也快
    for step in range(steps):
        tmp = out.detach()*(1-bg_m_t) + out*bg_m_t
        loss = cal_laplacian_loss(fg_t, fg_m_t, tmp, bg_m_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 300 == 0:
            print(f"step {step}, loss {loss.item():.4f}")
    res = torch.clamp(out, 0, 1).detach().cpu().squeeze().permute(1,2,0).numpy()*255
    return Image.fromarray(res.astype(np.uint8))

# 闭合+重置偏移
def close_and_reset(img, ps, dx, dy, bg_orig):
    img2, ps2 = close_polygon(img, ps)
    bg2 = update_background(bg_orig, ps2, 0, dy)
    return img2, ps2, bg2, gr.update(value=0)

# ------------------- Gradio UI -------------------
with gr.Blocks(title="Poisson Blending") as demo:
    poly_state = gr.State(initialize_polygon())
    bg_original = gr.State(None)
    gr.Markdown("# 泊松图像融合")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 前景图")
            fg_input = gr.Image(type="pil", height=300)
            gr.Markdown("### 绘制选区（点3个以上）")
            fg_show = gr.Image(type="pil", interactive=True, height=300)
            btn_close = gr.Button("闭合选区")
        with gr.Column():
            gr.Markdown("### 背景图")
            bg_input = gr.Image(type="pil", height=300)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 背景预览")
            bg_show = gr.Image(type="pil", height=400)
        with gr.Column():
            gr.Markdown("### 融合结果")
            output = gr.Image(type="pil", height=400)
    with gr.Row():
        dx = gr.Slider(-300,300,0, label="水平偏移")
        dy = gr.Slider(-300,300,0, label="垂直偏移")
        btn_blend = gr.Button("开始融合")
    # 事件绑定
    fg_input.change(lambda x:x, fg_input, fg_show)
    fg_show.select(add_point, [fg_input, poly_state], [fg_show, poly_state])
    btn_close.click(close_and_reset, [fg_input, poly_state, dx, dy, bg_original], [fg_show, poly_state, bg_show, dx])
    bg_input.change(lambda x:x, bg_input, bg_original)
    dx.change(update_background, [bg_original, poly_state, dx, dy], bg_show)
    dy.change(update_background, [bg_original, poly_state, dx, dy], bg_show)
    btn_blend.click(blending, [fg_input, bg_input, dx, dy, poly_state], output)

# 关键：让Gradio自动找可用端口，不强制7860
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=None, quiet=False)