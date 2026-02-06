import torch
from PIL import Image
import os


from amyself.model_new import Difix_my, load_model


CHECKPOINT_PATH = "outputs/13/checkpoints/model_2401.pkl"


# x_src in training
INPUT_IMAGE_PATH = "/path/to/your/source_image.png" 
# I_ref in training
REF_IMAGE_PATH = "/path/to/your/reference_image.png"
# I_before in training
BEFORE_IMAGE_PATH = "/path/to/your/before_image.png"
# I_after in training
AFTER_IMAGE_PATH = "/path/to/your/after_image.png"


PROMPT = "a high-quality photograph"


OUTPUT_PATH = "generated_output.png"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAINING_TIMESTEP = 199 

print(">>> [1/4] 正在加载模型...")


model = Difix_my(timestep=TRAINING_TIMESTEP)


model, _, _ = load_model(
    net_difix=model, 
    optimizer=None, 
    lr_scheduler=None, 
    pretrained_path=CHECKPOINT_PATH
)


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
    print(f"erro！ {e}")
    exit()

print(">>> data is already")

# ==============================================================================
# 4. 执行推理
# ==============================================================================

print(">>> [3/4] sampling")


with torch.no_grad():

    output_image_pil = model.sample(
        image=input_image,
        width=input_image.width,   
        height=input_image.height,
        ref_image=ref_image,
        before_image=before_image,
        after_image=after_image,
        prompt=PROMPT
    )

print(">>> complete")


print(">>> [4/4] saving")


output_image_pil.save(OUTPUT_PATH)

print(f"*** success ***")


