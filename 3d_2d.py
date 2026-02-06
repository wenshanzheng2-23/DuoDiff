# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
import json
from PIL import Image
from pathlib import Path

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server


# Helper functions for sky segmentation


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf

def save_camera_params(image_names, extrinsics, intrinsics, save_path="camera_params.jsonl"):
    """
    Args:
        image_names (list[str]): 输入的图片路径列表
        extrinsics (np.ndarray): (S, 3, 4) 相机外参
        intrinsics (np.ndarray): (S, 3, 3) 相机内参
        save_path (str): 输出的 JSON 文件路径
    """
    with open(save_path, "w", encoding="utf-8") as f:
        for img_path, ext, intr in zip(image_names, extrinsics, intrinsics):
            record = {
                "image_name": os.path.basename(img_path),
                "extrinsic": ext.tolist(),
                "intrinsic": intr.tolist()
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Camera parameters saved to {save_path}")


def load_cameras_from_jsonl(jsonl_path, default_hw=None):
    """
    读取 JSONL，每行包含: image_name, intrinsic(3x3), extrinsic(3x4)
    返回: list[dict]，每个包含: name, K, E, H, W
    说明:
      - 如果没给 default_hw，则用  H≈round(2*cy), W≈round(2*cx) 估一个分辨率
    """
    cams = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            K = np.array(rec["intrinsic"], dtype=np.float32)
            E = np.array(rec["extrinsic"], dtype=np.float32)  # world->cam, shape (3,4)

            if default_hw is not None:
                H, W = default_hw
            else:
                cx, cy = K[0, 2], K[1, 2]
                W = int(round(2.0 * cx))
                H = int(round(2.0 * cy))

            cams.append({
                "name": os.path.splitext(rec["image_name"])[0],
                "K": K,
                "E": E,
                "H": H,
                "W": W
            })
    return cams


def build_cameras_from_predictions(predictions, default_hw=None):
    """
    从 predictions 构造相机列表（与原 JSONL 的 cams 结构一致）
    predictions 需要包含：
      - predictions["intrinsic"]: (S,3,3)
      - predictions["extrinsic"]: (S,3,4)  # world->camera
      - predictions["image_names"]: list[str] 或 (S,) 的路径列表
      - 可选：predictions["images"]: (S,3,H,W) 以便推断 H、W
    default_hw:
      - 若给定 (H,W)，统一使用该分辨率；
      - 否则优先从 predictions["images"] 取 (H,W)；还没有就用 cx,cy 估算。
    返回：
      cams: list[dict]，每个包含 {name, K, E, H, W}
    """
    intrinsics = predictions["intrinsic"]   # (S,3,3)
    extrinsics = predictions["extrinsic"]   # (S,3,4) world->cam
    image_names = predictions.get("image_names", [f"frame_{i:05d}.png" for i in range(len(intrinsics))])

    cams = []
    S = intrinsics.shape[0]

    # 如果有 images，就用于推断统一分辨率
    images_tensor = predictions.get("images", None)
    if default_hw is None and images_tensor is not None:
        _, _, H_img, W_img = images_tensor.shape
        default_hw = (H_img, W_img)

    for i in range(S):
        K = intrinsics[i]
        E = extrinsics[i]  # world->camera
        if default_hw is not None:
            H, W = default_hw
        else:
            # 回退：根据 (cx,cy) 估算输出分辨率
            cx, cy = K[0, 2], K[1, 2]
            W = int(round(2.0 * cx))
            H = int(round(2.0 * cy))
        cams.append({
            "name": Path(image_names[i]).stem,
            "K": K.astype(np.float32),
            "E": E.astype(np.float32),
            "H": int(H),
            "W": int(W),
        })
    return cams



def project_pointcloud_to_image(points, colors, K, E, H, W, bg_value=0):
    """
    将世界坐标点云投影到图像平面 (稀疏渲染，Z-buffer 取最近点)
    Args:
        points: (N,3) 世界坐标
        colors: (N,3) 颜色; 可为 0~1 或 0~255
        K:      (3,3) 相机内参
        E:      (3,4) 相机外参 (world->camera) = [R|t]
        H,W:    输出图像尺寸
        bg_value: 背景值。可设为 0(黑) 或元组 (B,G,R)
    Returns:
        img: (H,W,3) uint8
        depth: (H,W) float32，未命中像素为 +inf
        mask: (H,W) bool，命中的像素为 True
    """
    # 规范化 colors 到 0~255 uint8
    col = np.asarray(colors, dtype=np.float32)
    if col.max() <= 1.0:
        col = (col * 255.0)
    col = np.clip(col, 0, 255).astype(np.uint8)

    N = points.shape[0]   # （N，3）
    homo = np.hstack([points, np.ones((N, 1), dtype=np.float32)])  # (N,4)
    # hstack 表示 在列方向拼接
    # 相机坐标 (N,3)
    cam_pts = (E @ homo.T).T   # (1*3)  由世界坐标系转换为相机坐标系
    z = cam_pts[:, 2]  # depth
    front_mask = z > 0  # 相机前方
    cam_pts = cam_pts[front_mask]
    col = col[front_mask]
    z = z[front_mask]   # 只保留相机前方的点

    # 像素坐标 (N,3) -> 归一化再取 u,v
    pixels = (K @ cam_pts.T).T
    pixels /= pixels[:, 2:3]

    # u = pixels[:, 0].astype(np.int32)
    # v = pixels[:, 1].astype(np.int32)
    u = np.rint(pixels[:, 0]).astype(np.int32)  # 四舍五入取整 比暴力取整效果好一些
    v = np.rint(pixels[:, 1]).astype(np.int32)

    # 仅保留画幅内的像素（如前面没做 z>0，这里可以一并加上 & (z > 0)）
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, col = u[ok], v[ok], z[ok], col[ok]


    # 初始化输出
    if isinstance(bg_value, (tuple, list, np.ndarray)):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :] = np.array(bg_value, dtype=np.uint8)
    else:
        img = np.full((H, W, 3), int(bg_value), dtype=np.uint8)

    depth = np.full((H, W), np.inf, dtype=np.float32)

    # 写像素：Z-buffer 取最近深度
    for x, y, zz, c in zip(u, v, z, col):
        if zz < depth[y, x]:
            img[y, x] = c
            # depth[y, x] = zz

    return img


def render_views_from_jsonl(jsonl_path, points, colors,h=None, w=None,
                            out_dir="reproj", bg_value=0, only_names=None):
    """
    批量渲染：从 JSONL 读取相机，投影点云并保存图片
    Args:
        jsonl_path: 相机参数 JSONL 路径
        points, colors: 点云与颜色 (世界坐标)
        default_hw: 指定统一 (H,W)。不指定则由 K 的 (cx,cy) 估算。
        out_dir: 输出目录
        bg_value: 背景像素值(空洞像素)，0/黑 或 (B,G,R)
        only_names: 只渲染给定的图名列表（不带扩展名），可为 None
    """
    os.makedirs(out_dir, exist_ok=True)
    cams = load_cameras_from_jsonl(jsonl_path, default_hw=(h, w))

    results = {}
    for cam in cams:
        if (only_names is not None) and (cam["name"] not in only_names):
            continue

        img = project_pointcloud_to_image(
            points, colors, cam["K"], cam["E"], cam["H"], cam["W"], bg_value = bg_value 
        )
        save_path = os.path.join(out_dir, f"{cam['name']}_b.png")
        # cv2.imwrite(save_path, img)
        # vggt加载图片使用的PIL库：RGB ，而opencv为：BGR
        cv2.imwrite(save_path, img[..., ::-1])   # RGB -> BGR，再保存
        results[cam["name"]] = {"image": img, "save_path": save_path}
        print(f"[√] saved: {save_path}")

    return results


def render_views_from_predictions(predictions, points, colors,
                                  h=None, w=None, out_dir="reproj",
                                  bg_value=0, only_names=None, exclude_names=None):
    os.makedirs(out_dir, exist_ok=True)
    default_hw = (h, w) if (h is not None and w is not None) else None
    cams = build_cameras_from_predictions(predictions, default_hw=default_hw)

    # 规范化名单到集合
    only_set = set(only_names) if only_names else None
    exclude_set = set(exclude_names) if exclude_names else set()

    results = {}
    for cam in cams:
        name = cam["name"]

        # 1) 若指定 only_names，则只渲染这些（优先级更高）
        if only_set is not None and name not in only_set:
            continue

        # 2) 否则默认全渲染，排除 exclude_names 里的
        if only_set is None and name in exclude_set:
            continue

        img = project_pointcloud_to_image(
            points, colors, cam["K"], cam["E"], cam["H"], cam["W"], bg_value=bg_value
        )
        save_path = os.path.join(out_dir, f"{name}.png")
        cv2.imwrite(save_path, img[..., ::-1])   # RGB->BGR
        results[name] = {"image": img, "save_path": save_path}
        print(f"[√] saved: {save_path}")

    return results



parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images"
)
parser.add_argument("--out_image", type=str, default="examples/output/", help="Path to folder containing images")
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument("--thr", type=float, default=1.0, help="Apply sky segmentation to filter out sky points")


def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. 模型加载 (通常不计入单次推断测试时间) ---
    print("Initializing and loading VGGT model...")
    model = VGGT()
    MODEL_PATH = "/home/yun/workspace/model/huggingface/VGGT-1B/model.pt"
    state_dict = torch.load(MODEL_PATH, map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    # ==================== 测试开始 ====================
    pipeline_start = time.time()

    # --- 2. 图片加载与预处理耗时 ---
    preprocess_start = time.time()
    image_names = sorted(glob.glob(os.path.join(args.image_folder, "*.png")))
    if not image_names:
        print("No images found!")
        return
        
    with Image.open(image_names[0]) as img:
        W0, H0 = img.size
    
    images = load_and_preprocess_images(image_names).to(device)
    H1, W1 = images.shape[-2:]
    preprocess_end = time.time()
    print(f"\n[Timer] Preprocessing ({len(image_names)} frames) took: {preprocess_end - preprocess_start:.4f}s")

    # --- 3. 模型推理耗时 ---
    inference_start = time.time()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    
    if device == "cuda":
        torch.cuda.synchronize()  # 确保 GPU 计算完成
    inference_end = time.time()
    print(f"[Timer] Model Inference took: {inference_end - inference_start:.4f}s")

    # --- 4. 后处理与数据准备 (Pose/PCL) ---
    post_start = time.time()
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions["image_names"] = image_names

    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    
    name_to_idx = {Path(p).stem: i for i, p in enumerate(image_names)}
    selected_names = ["frame_00001", "frame_00031"] # 确保你的文件名匹配
    sel_idx = [name_to_idx[n] for n in selected_names if n in name_to_idx]

    wp_all = predictions["world_points"]
    imgs_all = predictions["images"]
    conf_all = predictions["world_points_conf"]
    if conf_all.ndim == 4:
        conf_all = conf_all[:, 0]

    # 点云筛选逻辑
    wp = wp_all[sel_idx]
    imgs = imgs_all[sel_idx]
    conf_sel = conf_all[sel_idx]
    points = wp.reshape(-1, 3)
    colors = (imgs.transpose(0, 2, 3, 1).reshape(-1, 3) * 255.0).astype(np.uint8)

    mask = (conf_sel > args.thr).reshape(-1)
    valid_xyz = np.isfinite(wp).all(axis=-1).reshape(-1)
    final_mask = mask & valid_xyz
    points = points[final_mask]
    colors = colors[final_mask]
    post_end = time.time()
    print(f"[Timer] Post-processing & PCL extraction took: {post_end - post_start:.4f}s")

    # --- 5. 反投影渲染耗时 ---
    render_start = time.time()
    render_views_from_predictions(
        predictions,
        points,
        colors,
        h=H1,
        w=W1,
        out_dir=args.out_image,
        only_names=selected_names
    )
    render_end = time.time()
    print(f"[Timer] Reprojection Rendering took: {render_end - render_start:.4f}s")

    # ==================== 测试结束 ====================
    pipeline_total = time.time() - pipeline_start
    print("-" * 50)
    print(f"Total Pipeline Time: {pipeline_total:.4f}s")
    print(f"Average Inference FPS: {len(image_names) / (inference_end - inference_start):.2f}")
    print("-" * 50)

    # # 可视化部分（根据需要开启）
    # if args.port > 0:
    #     print("Starting viser visualization...")
    #     viser_wrapper(
    #         predictions,
    #         port=args.port,
    #         init_conf_threshold=args.conf_threshold,
    #         use_point_map=args.use_point_map,
    #         background_mode=args.background_mode,
    #         mask_sky=args.mask_sky,
    #         image_folder=args.image_folder,
    #     )

if __name__ == "__main__":
    main()
