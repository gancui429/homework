import cv2
import numpy as np
import gradio as gr

# =========================================================
# 全局变量
# =========================================================
image = None                 # 原图（RGB）
display_image = None         # 显示图（RGB）
display_scale = 1.0          # 显示图 / 原图 缩放比例

points_src = []              # 用户选择的源点（显示图坐标）
points_dst = []              # 用户选择的目标点（显示图坐标）

MAX_DISPLAY_SIDE = 700


# =========================================================
# 工具函数
# =========================================================
def resize_for_display(img, max_side=MAX_DISPLAY_SIDE):
    h, w = img.shape[:2]
    longest = max(h, w)

    if longest <= max_side:
        return img.copy(), 1.0

    scale = max_side / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def render_marked_image(base_img, src_pts, dst_pts):
    if base_img is None:
        return None

    marked = base_img.copy()

    pair_num = min(len(src_pts), len(dst_pts))

    # 已配对的点
    for i in range(pair_num):
        sx, sy = int(src_pts[i][0]), int(src_pts[i][1])
        tx, ty = int(dst_pts[i][0]), int(dst_pts[i][1])

        # 判断是不是固定点
        is_anchor = (sx == tx and sy == ty)

        if is_anchor:
            cv2.circle(marked, (sx, sy), 5, (255, 255, 0), -1)  # 青色固定点
            cv2.putText(
                marked, f"A{i+1}", (sx + 5, sy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA
            )
        else:
            cv2.circle(marked, (sx, sy), 4, (255, 0, 0), -1)    # 蓝色源点
            cv2.circle(marked, (tx, ty), 4, (0, 0, 255), -1)    # 红色目标点

            cv2.putText(
                marked, f"S{i+1}", (sx + 5, sy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
            )
            cv2.putText(
                marked, f"T{i+1}", (tx + 5, ty - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
            )

            cv2.arrowedLine(marked, (sx, sy), (tx, ty), (0, 255, 0), 2, tipLength=0.2)

    # 如果最后还剩一个未配对源点
    if len(src_pts) > len(dst_pts):
        sx, sy = int(src_pts[-1][0]), int(src_pts[-1][1])
        cv2.circle(marked, (sx, sy), 4, (255, 0, 0), -1)
        cv2.putText(
            marked, f"S{len(src_pts)}", (sx + 5, sy - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
        )

    return marked


def build_status_text():
    if image is None:
        return "请先上传图片。"

    pair_num = min(len(points_src), len(points_dst))
    anchor_num = 0
    for i in range(pair_num):
        if points_src[i][0] == points_dst[i][0] and points_src[i][1] == points_dst[i][1]:
            anchor_num += 1

    if len(points_src) == len(points_dst):
        return f"当前共有 {pair_num} 对点，其中固定点 {anchor_num} 对。下一次点击请选择新的源点。"
    else:
        return f"当前共有 {pair_num} 对完整点，正在等待第 {len(points_src)} 个源点对应的目标点。"


def add_boundary_anchor_points(src, dst, h, w):
    """
    自动添加边界固定点，防止整体塌陷或条纹
    """
    anchors = [
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w // 2, 0],
        [w // 2, h - 1],
        [0, h // 2],
        [w - 1, h // 2],
    ]

    if len(src) == 0:
        src_aug = np.array(anchors, dtype=np.float32)
        dst_aug = np.array(anchors, dtype=np.float32)
    else:
        src_aug = np.vstack([src, np.array(anchors, dtype=np.float32)])
        dst_aug = np.vstack([dst, np.array(anchors, dtype=np.float32)])

    return src_aug, dst_aug


# =========================================================
# 核心算法：MLS仿射变形
# =========================================================
def point_guided_deformation(img, src_points, dst_points, fixed_points=None, alpha=1.0):
    """
    点引导的MLS图像变形

    参数:
        img: 输入图像 (numpy数组)
        src_points: 源控制点列表，格式[[x1,y1], [x2,y2], ...]
        dst_points: 目标控制点列表，格式[[x1,y1], [x2,y2], ...]
        fixed_points: 固定点列表，格式[[x1,y1], [x2,y2], ...] (可选)
        alpha: 权重平滑度参数，控制变形的局部性

    返回:
        变形后的图像
    """
    # 合并固定点到控制点列表（固定点等价于 src=dst 的约束点）
    if fixed_points is not None and len(fixed_points) > 0:
        src_points = src_points + fixed_points
        dst_points = dst_points + fixed_points

    return mls_affine_warp(img, src_points, dst_points, alpha=alpha)


def mls_affine_warp(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    MLS仿射变形核心实现

    算法流程:
    1. 对输出图像中的每个像素v，计算其到所有控制点的权重
    2. 计算加权重心 p* 和 q*
    3. 构建局部仿射变换矩阵 M = inv(C) @ B
    4. 计算源图像中的对应位置 f(v) = (v - p*) M + q*
    5. 使用逆向映射对图像进行采样

    参数:
        image: 输入图像
        source_pts: 源控制点 q_i
        target_pts: 目标控制点 p_i
        alpha: 权重平滑度参数
        eps: 数值稳定性小常数

    返回:
        变形后的图像
    """
    img = np.asarray(image, dtype=np.uint8)
    h, w = img.shape[:2]

    src = np.asarray(source_pts, dtype=np.float32)   # q_i
    dst = np.asarray(target_pts, dtype=np.float32)   # p_i

    if len(src) < 1 or len(src) != len(dst):
        return img.copy()

    # 自动加边界固定点防止整体塌陷
    src, dst = add_boundary_anchor_points(src, dst, h, w)

    n = len(src)

    # 输出图像素网格 v
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    V = np.stack([grid_x, grid_y], axis=-1)  # (h, w, 2)

    # diff = v - p_i
    diff = V[:, :, None, :] - dst[None, None, :, :]   # (h,w,n,2)
    d2 = np.sum(diff * diff, axis=-1)                 # (h,w,n)

    # 哪些点刚好等于某个目标控制点
    exact_match = d2 < eps

    # 权重 w_i(v) = 1/||v-p_i||^{2α}
    weights = 1.0 / np.power(d2 + eps, alpha)         # (h,w,n)
    weights_sum = np.sum(weights, axis=-1, keepdims=True)

    # 加权中心 p* 和 q*
    p_star = np.sum(weights[..., None] * dst[None, None, :, :], axis=2) / weights_sum
    q_star = np.sum(weights[..., None] * src[None, None, :, :], axis=2) / weights_sum

    # 去中心化
    phat = dst[None, None, :, :] - p_star[:, :, None, :]   # (h,w,n,2)
    qhat = src[None, None, :, :] - q_star[:, :, None, :]   # (h,w,n,2)

    # C = sum_i w_i * phat_i * phat_i^T
    C = np.einsum("hwn,hwna,hwnb->hwab", weights, phat, phat)  # (h,w,2,2)

    # B = sum_i w_i * phat_i * qhat_i^T
    B = np.einsum("hwn,hwna,hwnb->hwab", weights, phat, qhat)  # (h,w,2,2)

    # 数值稳定
    eye = np.eye(2, dtype=np.float32)[None, None, :, :]
    C = C + eps * eye

    # M = inv(C) @ B
    C_inv = np.linalg.inv(C)
    M = np.einsum("hwab,hwbc->hwac", C_inv, B)

    # f(v) = (v - p*) M + q*
    v_minus_pstar = V - p_star
    Q = np.einsum("hwa,hwab->hwb", v_minus_pstar, M) + q_star

    # 精确控制点强制匹配
    for i in range(n):
        mask = exact_match[:, :, i]
        Q[mask] = src[i]

    map_x = Q[:, :, 0].astype(np.float32)
    map_y = Q[:, :, 1].astype(np.float32)

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    return warped


# =========================================================
# 事件函数
# =========================================================
def upload_image(img):
    global image, display_image, display_scale, points_src, points_dst

    points_src.clear()
    points_dst.clear()

    if img is None:
        image = None
        display_image = None
        display_scale = 1.0
        return None, None, "请先上传图片。"

    image = img.copy()
    display_image, display_scale = resize_for_display(image)

    return img, display_image.copy(), (
        "图片上传成功。"
        "在左下图中按 源点 → 目标点 的顺序点击。"
        "如果要加固定点，请在同一个位置连续点两次。"
    )


def record_points(evt: gr.SelectData):
    global points_src, points_dst, display_image

    if display_image is None:
        return None, "当前没有图片，请先上传。"

    x, y = int(evt.index[0]), int(evt.index[1])

    if len(points_src) == len(points_dst):
        points_src.append([x, y])
        status = f"已选择第 {len(points_src)} 个源点，请点击对应目标点。固定点请在同一位置再点一次。"
    else:
        points_dst.append([x, y])
        status = build_status_text()

    marked = render_marked_image(display_image, points_src, points_dst)
    return marked, status


def run_warping(alpha=1.0):
    global image, display_scale, display_image

    if image is None:
        return None, "请先上传图片。"

    if len(points_src) == 0 or len(points_src) != len(points_dst):
        return None, "请至少选择一对完整的点。"

    # 显示图坐标 -> 原图坐标
    src_orig = np.asarray(points_src, dtype=np.float32) / display_scale
    dst_orig = np.asarray(points_dst, dtype=np.float32) / display_scale

    warped = point_guided_deformation(
        image,
        src_orig,
        dst_orig,
        fixed_points=None,
        alpha=alpha
    )

    # 为了网页显示不"看起来被裁切"，把结果缩放到和显示图一致
    show_h, show_w = display_image.shape[:2]
    warped_show = cv2.resize(warped, (show_w, show_h), interpolation=cv2.INTER_LINEAR)

    return warped_show, f"变形完成，共使用 {len(points_src)} 对点（包含固定点）。"


def clear_points():
    global points_src, points_dst, display_image

    points_src.clear()
    points_dst.clear()

    if display_image is None:
        return None, None, "当前没有图片可清空。"

    return display_image.copy(), None, "控制点已清空，请重新选择。"


# =========================================================
# 界面
# =========================================================
with gr.Blocks() as demo:
    gr.Markdown("## 图像拉伸变形（MLS Moving Least Squares）")
    gr.Markdown(
        "先上传图片，再在左下图中按 **源点 → 目标点** 的顺序依次点击。"
        "如果要添加固定点，请在同一个位置连续点击两次。"
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="上传图片",
                interactive=True,
                type="numpy",
                height=220
            )

            point_select = gr.Image(
                label="点击选择源点和目标点",
                interactive=True,
                type="numpy",
                height=500
            )

        with gr.Column():
            result_image = gr.Image(
                label="变形结果",
                type="numpy",
                height=500
            )

    status_box = gr.Textbox(
        label="当前状态",
        value="请先上传图片。",
        interactive=False
    )

    alpha_slider = gr.Slider(
        minimum=0.1,
        maximum=2.0,
        value=1.0,
        step=0.1,
        label="Alpha 参数 (权重平滑度)"
    )

    with gr.Row():
        run_button = gr.Button("Run Warping")
        clear_button = gr.Button("Clear Points")

    input_image.upload(
        upload_image,
        inputs=input_image,
        outputs=[input_image, point_select, status_box]
    )

    point_select.select(
        record_points,
        inputs=None,
        outputs=[point_select, status_box]
    )

    run_button.click(
        run_warping,
        inputs=[alpha_slider],
        outputs=[result_image, status_box]
    )

    clear_button.click(
        clear_points,
        inputs=None,
        outputs=[point_select, result_image, status_box]
    )

demo.launch(share=True)