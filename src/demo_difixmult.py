import os
import json
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import Difix
from pathlib import Path # <--- 1. 引入 pathlib

# ==============================================================================
# 1. 在这里配置您的所有路径和参数
# ==============================================================================
CONFIG = {
    "json_path": "data/DL3DV3_38/twenty.json",              
    "model_path": "/media/yun/usr/huggingface/difix/",   
    "output_dir": "/media/yun/usr/difix_out/twenty",                 
    "dataset_to_process": "train", 
    "use_ref_image": True,
    "height": 576,
    "width": 1024,
    "timestep": 199,
    "seed": 42,
    # "save_as_video": False,  <--- 此功能与分场景保存冲突，已移除
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
    
    # 检查基本配置
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found at: {model_path}")

    # 主输出目录现在会在循环内按需创建，这里仅打印信息
    print(f"Outputs will be saved within: {os.path.abspath(output_dir)}")

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

    items_to_process = data.get(dataset_to_process, {})
    if not items_to_process:
        print(f"No data found for dataset '{dataset_to_process}' in the JSON file. Exiting.")
        return

    print(f"Found {len(items_to_process)} items to process in '{dataset_to_process}' dataset.")
    
    # ---【核心修改】将处理和保存合并到同一个循环中 ---
    for item_id, item_data in tqdm(items_to_process.items(), desc="Processing and saving images"):
        try:
            input_image_path = item_data.get('image')
            ref_image_path = item_data.get('ref_image') if use_ref_image else None
            prompt = item_data.get('prompt')

            if not input_image_path or not os.path.exists(input_image_path):
                print(f"\nWarning: Skipping item '{item_id}' due to missing input image path.")
                continue
            if use_ref_image and (not ref_image_path or not os.path.exists(ref_image_path)):
                print(f"\nWarning: Skipping item '{item_id}' due to missing reference image path.")
                continue
            
            image = Image.open(input_image_path).convert('RGB')
            ref_image = Image.open(ref_image_path).convert('RGB') if ref_image_path else None

            output_image = model.sample(
                image,
                height=height,
                width=width,
                ref_image=ref_image,
                prompt=prompt if prompt else "A high-quality, photorealistic image."
            )

            # ---【核心修改】动态构建输出路径并立即保存 ---
            p = Path(input_image_path)
            
            # 提取场景文件夹名 (路径中的倒数第三个部分)
            scene_folder_name = p.parts[-3] 
            
            # 构建并创建该场景的输出目录
            scene_output_dir = Path(output_dir) / scene_folder_name
            scene_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 构建最终输出文件路径并保存
            save_path = scene_output_dir / p.name # p.name 是原始文件名
            output_image.save(save_path)

        except Exception as e:
            import traceback
            print(f"\nError processing item '{item_id}': {e}")
            traceback.print_exc()

    print(f"\nProcessing complete. All images saved in their respective scene folders within {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
