import torch
from PIL import Image
import os

# 确保模型定义代码可用
# 'model_new.py' 应与此脚本在同一目录，或者在Python的搜索路径中
from amyself.model_new import Difix_my, load_model

# ==============================================================================
# 1. 配置参数 (请根据您的实际情况修改)
# ==============================================================================

# 指向您训练好的模型权重文件 (.pkl 或 .pt)
CHECKPOINT_PATH = "outputs/13/checkpoints/model_2401.pkl"

# 输入图像路径
# x_src in training
INPUT_IMAGE_PATH = "/path/to/your/source_image.png" 
# I_ref in training
REF_IMAGE_PATH = "/path/to/your/reference_image.png"
# I_before in training
BEFORE_IMAGE_PATH = "/path/to/your/before_image.png"
# I_after in training
AFTER_IMAGE_PATH = "/path/to/your/after_image.png"

# 文本提示
PROMPT = "a high-quality photograph" # 使用您训练时采用的或期望的提示

# 输出图像保存路径
OUTPUT_PATH = "generated_output.png"

# 使用的设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 实例化模型时使用的timestep，应与训练时保持一致
# 根据您的训练脚本，默认为 199
TRAINING_TIMESTEP = 199 

# ==============================================================================
# 2. 加载模型
# ==============================================================================

print(">>> [1/4] 正在加载模型...")

# 实例化模型
# 注意：这里的参数应与您训练时使用的模型配置相匹配
model = Difix_my(timestep=TRAINING_TIMESTEP)

# 加载预训练权重
# 对于推理，optimizer 和 lr_scheduler 设置为 None
model, _, _ = load_model(
    net_difix=model, 
    optimizer=None, 
    lr_scheduler=None, 
    pretrained_path=CHECKPOINT_PATH
)

# 设置为评估模式并移动到GPU
model.set_eval()
model.to(DEVICE)

print(f">>> 模型加载完毕，已切换至 {DEVICE} 评估模式。")

# ==============================================================================
# 3. 准备输入数据
# ==============================================================================

print(">>> [2/4] 正在加载并准备输入图像...")

# 使用PIL加载所有需要的图像
try:
    input_image   = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    ref_image     = Image.open(REF_IMAGE_PATH).convert("RGB")
    before_image  = Image.open(BEFORE_IMAGE_PATH).convert("RGB")
    after_image   = Image.open(AFTER_IMAGE_PATH).convert("RGB")
except FileNotFoundError as e:
    print(f"错误：找不到图像文件！ {e}")
    exit()

print(">>> 输入数据准备就绪。")

# ==============================================================================
# 4. 执行推理
# ==============================================================================

print(">>> [3/4] 正在执行模型推理...")

# 推理过程不计算梯度
with torch.no_grad():
    # 直接调用封装好的sample方法
    # 它内部处理了所有的图像变换和批处理
    output_image_pil = model.sample(
        image=input_image,
        width=input_image.width,     # 保持原始图像尺寸
        height=input_image.height,
        ref_image=ref_image,
        before_image=before_image,
        after_image=after_image,
        prompt=PROMPT
    )

print(">>> 推理完成。")

# ==============================================================================
# 5. 保存结果
# ==============================================================================

print(">>> [4/4] 正在保存输出图像...")

# 保存生成的PIL Image对象
output_image_pil.save(OUTPUT_PATH)

print(f"*** 成功！图像已保存至: {os.path.abspath(OUTPUT_PATH)} ***")

# (可选) 如果在支持GUI的环境中运行，可以取消注释以直接显示图片
# output_image_pil.show()
