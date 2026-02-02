import os
import json
import imageio
# import argparse # 不再需要命令行参数
import numpy as np
from PIL import Image
# from glob import glob # 不再需要
from tqdm import tqdm
from model import Difix

# ==============================================================================
# 1. 在这里配置您的所有路径和参数
# ==============================================================================
CONFIG = {
    # --- 必须配置的路径 ---
    "json_path": "data/DL3DV3_38/twenty.json",              
    "model_path": "/media/yun/usr/huggingface/difix/",   
    "output_dir": "/media/yun/usr/difix_out/twenty",                 

    "dataset_to_process": "train", 
    "use_ref_image": True,          # 如果JSON中有'ref_image'并且想使用它，设为True
    "height": 576,
    "width": 1024,
    "timestep": 199,
    "seed": 42,
    "save_as_video": False,         # 如果想保存为视频，设为True
}
# ==============================================================================

def main():
    # 从 CONFIG 字典中获取配置
    json_path = CONFIG["json_path"]
    model_path = CONFIG["model_path"]
    output_dir = CONFIG["output_dir"]
    dataset_to_process = CONFIG["dataset_to_process"]
    use_ref_image = CONFIG["use_ref_image"]
    height = CONFIG["height"]
    width = CONFIG["width"]
    timestep = CONFIG["timestep"]
    save_as_video = CONFIG["save_as_video"]

    # 检查基本配置
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found at: {model_path}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")

    # 初始化模型 (只执行一次)
    print("Initializing the Difix model...")
    model = Difix(
        pretrained_path=model_path,
        timestep=timestep,
        mv_unet=use_ref_image,
    )
    model.set_eval()
    print("Model initialized successfully.")

    # 读取并解析 JSON 文件
    print(f"Loading data from JSON file: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 根据配置选择要处理的数据集
    items_to_process = {}
    if dataset_to_process in ['train', 'all']:
        items_to_process.update(data.get('train', {}))
    if dataset_to_process in ['test', 'all']:
        items_to_process.update(data.get('test', {}))

    if not items_to_process:
        print(f"No data found for dataset '{dataset_to_process}' in the JSON file. Exiting.")
        return

    print(f"Found {len(items_to_process)} items to process in '{dataset_to_process}' dataset(s).")
    
    # 核心处理循环
    output_images = []
    output_filenames = []
    
    for item_id, item_data in tqdm(items_to_process.items(), desc="Processing images"):
        input_image_path = item_data.get('image')
        ref_image_path = item_data.get('ref_image') if use_ref_image else None
        prompt = item_data.get('prompt')

        # 安全检查
        if not input_image_path or not os.path.exists(input_image_path):
            print(f"\nWarning: Skipping item '{item_id}' due to missing or invalid input image path: {input_image_path}")
            continue
        if use_ref_image and (not ref_image_path or not os.path.exists(ref_image_path)):
            print(f"\nWarning: Skipping item '{item_id}' due to missing or invalid reference image path: {ref_image_path}")
            continue
        
        # 打开图片
        image = Image.open(input_image_path).convert('RGB')
        ref_image = Image.open(ref_image_path).convert('RGB') if ref_image_path else None

        # 执行推理
        output_image = model.sample(
            image,
            height=height,
            width=width,
            ref_image=ref_image,
            prompt=prompt if prompt else "A high-quality, photorealistic image."
        )
        output_images.append(output_image)
        
        # 2. 文件名保存逻辑: 与原图片文件名一致
        # 从 "image" 字段的完整路径中提取文件名 (例如: "frame_00001.png")
        original_filename = os.path.basename(input_image_path)
        output_filenames.append(original_filename)

    # 保存逻辑
    if save_as_video:
        print("Saving outputs as a video...")
        video_path = os.path.join(output_dir, "output.mp4")
        writer = imageio.get_writer(video_path, fps=30)
        for output_image in tqdm(output_images, desc="Saving video"):
            writer.append_data(np.array(output_image))
        writer.close()
        print(f"Video saved to {video_path}")
    else:
        print("Saving outputs as individual images...")
        for i, output_image in enumerate(tqdm(output_images, desc="Saving images")):
            # 使用提取出的原始文件名进行保存
            save_path = os.path.join(output_dir, output_filenames[i])
            output_image.save(save_path)
        print(f"All images saved in {output_dir}")

if __name__ == "__main__":
    main()
