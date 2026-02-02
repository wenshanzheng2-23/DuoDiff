import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from DEC import DECNet
from torch.utils.checkpoint import checkpoint

# =============== 你已有的调度/vae fwd ===============
def make_sched(timestep=199):
    sched = DDPMScheduler.from_pretrained("/home/yun/workspace/model/huggingface/sd-turbo", subfolder="scheduler")
    sched.set_timesteps(1, device="cuda")
    # sched.timesteps = torch.tensor([timestep], device="cuda")
    sched.alphas_cumprod = sched.alphas_cumprod.cuda()
    return sched

def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample

def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    for up_block in self.up_blocks:
        sample = up_block(sample, latent_embeds)
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def save_model(net_difix, optimizer, lr_scheduler, outf):
    """
    保存所有可训练/可配置部分的权重，包括 DEC, 注入模块, 门控参数等。
    
    Args:
        net_difix (Difix_my): 你的模型实例。
        optimizer (torch.optim.Optimizer): 包含所有可训练参数的优化器。
        outf (str): 输出文件路径 (.pt)。
    """
    if net_difix.dec is None and not any(p.requires_grad for p in net_difix.parameters()):
        print("Warning: DEC is None and no parameters require gradients. Nothing to save.")
        return
    # 准备一个字典来保存所有需要持久化的状态
    pkg = {
        "meta": {
            "unet_block_out_channels": tuple(net_difix.unet.config.block_out_channels),
        }
    }
    # 1. 保存 DECNet 的状态
    if net_difix.dec is not None:
        pkg["dec_state_dict"] = net_difix.dec.state_dict()
        # 顺便在 meta 中保存 DEC 的配置信息
        pkg["meta"]["dec_enc_channels"] = getattr(net_difix.dec, "enc_content", None) and tuple(net_difix.dec.enc_content.out_channels)
        pkg["meta"]["dec_fusion_type"] = type(net_difix.dec.fuses[0]).__name__ if len(net_difix.dec.fuses) > 0 else None
    # 2. 保存注入注意力模块 (InjectXAttnBlock) 的状态
    if net_difix.inj_xattn is not None:
        pkg["inj_xattn_state_dict"] = net_difix.inj_xattn.state_dict()

    # 保存优化器状态
    if optimizer is not None:
        pkg["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None:        
        pkg["lr_scheduler"] = lr_scheduler.state_dict()
    # print(f"Saving checkpoint with keys: {list(pkg.keys())}")
    torch.save(pkg, outf)
    # print(f"Model saved to {outf}")

def load_model(net_difix, optimizer, lr_scheduler, pretrained_path, new=None):

    print(f"Loading model from: {pretrained_path}")
    pkg = torch.load(pretrained_path, map_location="cpu")
    # 1. 加载 DECNet 的权重
    if "dec_state_dict" in pkg and net_difix.dec is not None:
        net_difix.dec.load_state_dict(pkg["dec_state_dict"])
        print(" -> Loaded 'dec_state_dict'.")
    else:
        print(" -> Warning: 'dec_state_dict' not found in checkpoint or model has no DEC module.")
    # 2. 加载注入注意力模块的权重
    if "inj_xattn_state_dict" in pkg and net_difix.inj_xattn is not None:
        net_difix.inj_xattn.load_state_dict(pkg["inj_xattn_state_dict"])
        print(" -> Loaded 'inj_xattn_state_dict'.")
    else:
        print(" -> Warning: 'inj_xattn_state_dict' not found in checkpoint.")
       
    # 打印一些元数据进行校对
    if "meta" in pkg and "unet_block_out_channels" in pkg["meta"]:
        print(f" -> Checkpoint's UNet block out channels: {pkg['meta']['unet_block_out_channels']}")
    # 6. 加载优化器状态
    if "optimizer" in pkg and optimizer is not None:
        try:
            optimizer.load_state_dict(pkg["optimizer"])
            print(" -> Loaded 'optimizer' state.")
            if new is not None:
                print(f" -> Manually overriding optimizer LR to {new}")
                for param_group in optimizer.param_groups:
                    print(f"上一个学习率{param_group['lr']}，新的学习率{new}")
                    print("*"*50)
                    param_group['lr'] = new
            else:
                for param_group in optimizer.param_groups:
                    print(f"上一个学习率{param_group['lr']}")
                    print("*"*50)
        except ValueError as e:
            print(f" -> Warning: Could not load optimizer state. It might be from a different model structure. Error: {e}")
    print("Model loading complete.")
    if "lr_scheduler" in pkg and lr_scheduler is not None:
        lr_scheduler.load_state_dict(pkg["lr_scheduler"])
    return net_difix, optimizer, lr_scheduler
    
class InjectXAttnBlock(nn.Module):
    """
    用 inj 作为条件，对 x 做一次 cross-attn + MLP 的 Transformer 注入。
    输入输出: [B,C,H,W] -> [B,C,H,W]（保持通道与空间尺寸不变）
    """
    def __init__(self, channels: int, n_heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        assert channels % n_heads == 0, f"channels({channels}) must be divisible by n_heads({n_heads})"
        self.norm_q = nn.GroupNorm(32, channels)
        self.norm_kv = nn.GroupNorm(32, channels)

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=n_heads, batch_first=True)

        self.out_proj = nn.Conv2d(channels, channels, 1)  # attn后的投影
        # FFN
        hidden = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x, inj):
        """
        x:   [B,C,H,W]
        inj: [B,C,H,W]（已对齐到相同空间尺寸与通道数）
        """
        B, C, H, W = x.shape

        # 归一化 + 线性投影到 Q/K/V
        q = self.q_proj(self.norm_q(x))          # [B,C,H,W]
        k = self.k_proj(self.norm_kv(inj))       # [B,C,H,W]
        v = self.v_proj(self.norm_kv(inj))       # [B,C,H,W]

        # 变成序列: [B, HW, C]
        q = q.flatten(2).transpose(1, 2)         # [B, HW, C]
        k = k.flatten(2).transpose(1, 2)         # [B, HW, C]
        v = v.flatten(2).transpose(1, 2)         # [B, HW, C]

        # cross-attn（Q 来自 x，KV 来自 inj）
        y, _ = self.attn(q, k, v, need_weights=False)  # [B, HW, C]

        # 回到 [B,C,H,W]
        y = y.transpose(1, 2).reshape(B, C, H, W)
        y = self.out_proj(y)

        # 残差1：attn 残差
        x = x + y

        # 残差2：MLP 残差
        x = x + self.ffn(x)
        return x
    
class InjectIdentity(nn.Module):
    """与 InjectXAttnBlock 接口一致，但直接返回 x，不做任何计算。"""
    def __init__(self): super().__init__()
    def forward(self, x, inj): return x


class Difix_my(nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None,
                 ckpt_folder="checkpoints", timestep=999,
                 use_dec=True,
                 dec_enc_channels=(320,640,1280,1280)):
        super().__init__()
        
        # 文本与调度
        self.tokenizer = AutoTokenizer.from_pretrained("/home/yun/workspace/model/huggingface/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("/home/yun/workspace/model/huggingface/sd-turbo", subfolder="text_encoder").cuda()
        self.text_encoder.requires_grad_(False)
        # __init__
        self.timestep = timestep                   # 先存成标量
        self.sched = make_sched(self.timestep)            # 再用它创建 scheduler
        self.timesteps = torch.tensor([self.timestep], device="cuda", dtype=torch.long)


        # VAE
        vae = AutoencoderKL.from_pretrained("/home/yun/workspace/model/huggingface/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

        unet = UNet2DConditionModel.from_pretrained("/home/yun/workspace/model/huggingface/sd-turbo", subfolder="unet")
        # unet = UNet2DConditionModel.from_config("/home/yun/workspace/model/huggingface/sd-turbo", subfolder="unet")

        # 预训练权重
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            _sd_unet = unet.state_dict()
            for k in sd.get("state_dict_unet", {}):
                if k in _sd_unet: _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            vae.encoder.requires_grad_(False)

        # 设备
        self.unet, self.vae = unet.to("cuda"), vae.to("cuda")
        # self.timesteps = torch.tensor([timestep], device="cuda").long()

        # ====== DEC：直接内嵌在 Difix ======
        self.use_dec = use_dec
        if self.use_dec:
            unet_chs = list(self.unet.config.block_out_channels)
            self.dec = DECNet(enc_channels=dec_enc_channels,
                        unet_block_out_channels=unet_chs,
                        fusion_type="concat_1x1",   # cross_attn
                        norm="gn", act="silu",
                        ds_mode="blurpool",
                        stage_depth=2,
                        proj_init="zero", proj_std=1e-3).cuda()
            self.dec.to("cuda")
            # 冻结原 UNet，全训 DEC（默认做法）
            for p in self.unet.parameters(): 
                p.requires_grad_(False)
        else:
            self.dec = None  # 不用 DEC，行为与原来一致
        # self.dec.to("cuda")
        # self.inj_xattn = nn.ModuleList(
        #     [InjectXAttnBlock(ch, n_heads=8, mlp_ratio=2.0) for ch in self.unet.config.block_out_channels]
        # )
        # 仅在低分辨率层（前两层）做注入；后两层禁用注入（使用恒等）
        # self.inj_xattn = nn.ModuleList([
        #     InjectXAttnBlock(unet_chs[0], n_heads=4, mlp_ratio=1.5),  # 320
        #     InjectXAttnBlock(unet_chs[1], n_heads=4, mlp_ratio=1.5),  # 640
        #     InjectIdentity(),  # 1280 层：禁用注入
        #     InjectIdentity(),  # 1280 层：禁用注入
        # ])
        self.inj_xattn = nn.ModuleList([
            InjectXAttnBlock(unet_chs[3], n_heads=4, mlp_ratio=1.5),  # 1280
            InjectXAttnBlock(unet_chs[2], n_heads=4, mlp_ratio=1.5),  # 1280
            InjectXAttnBlock(unet_chs[1], n_heads=4, mlp_ratio=1.5),  # 640
            InjectXAttnBlock(unet_chs[0], n_heads=4, mlp_ratio=1.5),  # 320
        ])
        # self.dec_gates = nn.Parameter(torch.ones(self.dec.levels) * 0.05)   # 初值 0.1
        # # self.dec_gates = nn.Parameter(torch.zeros(self.dec.levels))
        # self.inj_norms = nn.ModuleList([nn.GroupNorm(32, ch) for ch in self.unet.config.block_out_channels])

        # self.dec_adapters = nn.ModuleList([
        #     nn.Conv2d(C, C, 1, bias=True) for C in self.unet.config.block_out_channels
        # ])
        # for conv in self.dec_adapters:
        #     nn.init.zeros_(conv.weight); nn.init.zeros_(conv.bias)  # 初始相当于不加

        self.vae.requires_grad_(False)
        # 打印可训练参数量
        def _count(p): return sum(t.numel() for t in p if t.requires_grad)/1e6
        print("="*50)
        print(f"Trainable params — UNet: {_count(self.unet.parameters()):.2f}M")
        print(f"Trainable params — VAE:  {_count(self.vae.parameters()):.2f}M")
        print(f"Trainable params — DEC:  {(_count(self.dec.parameters()) if self.dec else 0.0):.2f}M")

        print("="*50)

    # ============= 公开接口：与原来保持兼容 =============
    def set_eval(self):
        self.unet.eval();
        self.vae.eval()
        self.unet.requires_grad_(False); 
        self.vae.requires_grad_(False)
        if self.dec is not None: self.dec.eval()

    def set_train(self, mode: bool = True):
        # —— 冻结 & eval：UNet / VAE / 文本编码器 ——
        self.unet.eval()
        self.vae.eval()
        if hasattr(self, "text_encoder"):
            self.text_encoder.eval()

        for p in self.unet.parameters(): p.requires_grad_(False)
        for p in self.vae.parameters():  p.requires_grad_(False)
        if hasattr(self, "text_encoder"):
            for p in self.text_encoder.parameters(): p.requires_grad_(False)

        # —— 训练：DEC + 注入相关模块 ——
        if self.dec is not None:
            self.dec.train()
            for p in self.dec.parameters(): p.requires_grad_(True)

        # 注入块（前两层是 InjectXAttnBlock，有参数；后两层 InjectIdentity，无参数也没关系）
        for blk in self.inj_xattn:
            blk.train()
            for p in blk.parameters(): p.requires_grad_(True)

        # # # 门控参数
        # self.dec_gates.requires_grad_(False)

        # # 注入归一化（GroupNorm：train/eval 影响不大，但统一为 train）
        # self.inj_norms.train()
        # for p in self.inj_norms.parameters(): p.requires_grad_(False)

        # # 1×1 适配器（ZeroConv/Adapter）
        # self.dec_adapters.train()
        # for p in self.dec_adapters.parameters(): p.requires_grad_(False)



    # ============= 核心前向 =============
    @torch.no_grad()
    def _encode_text(self, prompt=None, prompt_tokens=None):
        assert (prompt is None) != (prompt_tokens is None)
        if prompt is not None:
            toks = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                  padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            return self.text_encoder(toks)[0]
        else:
            return self.text_encoder(prompt_tokens)[0]
        
    def _unet_with_dec_forward_down(
        self,
        sample: torch.FloatTensor,           # 对应 z_noisy
        timestep: torch.FloatTensor,         # 对应 t
        encoder_hidden_states: torch.Tensor,
        I_ref: torch.FloatTensor = None,
        I_before: torch.FloatTensor = None,
        I_after: torch.FloatTensor = None,
        # 以下参数为保持与原始 forward 签名兼容，但根据您的 config，大部分不会被用到
        attention_mask: torch.Tensor = None,
        cross_attention_kwargs: dict = None,
        encoder_attention_mask: torch.Tensor = None,
    ):
        """
        根据您提供的 config.json 定制的 UNet 手动前向传播函数。
        如果提供了 I_ref/I_before/I_after，则启用 DEC 注入；否则走原 UNet。
        """
        # 检查是否启用 DEC 注入
        use_dec_injection = self.use_dec and (I_ref is not None and I_before is not None and I_after is not None)

        if not use_dec_injection:
            # 走原生 diffusers 前向，确保行为 100% 一致。
            # 这里只传入您的模型实际会用到的参数。
            return self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            ).sample

        # --- 手动前向传播，带 DEC 注入 ---

        # 1. 准备 DEC 注入特征
        with torch.no_grad():
            z_ref    = self.vae.encode(I_ref).latent_dist.mode()    * self.vae.config.scaling_factor
            z_before = self.vae.encode(I_before).latent_dist.mode() * self.vae.config.scaling_factor
            z_after  = self.vae.encode(I_after).latent_dist.mode()  * self.vae.config.scaling_factor
        injections = self.dec(z_ref, z_before, z_after)

        # 2. 计算时间嵌入 (根据您的 config 精简后的逻辑)
        # 确保 timestep 是 tensor 且与 sample 设备一致
        if not torch.is_tensor(timestep):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timestep = torch.tensor([timestep], dtype=dtype, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        
        # 广播到 batch 维度
        timesteps = timestep.expand(sample.shape[0])
        
        # 核心步骤： time_proj -> time_embedding
        t_emb = self.unet.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.unet.time_embedding(t_emb) # 您的模型没有 timestep_cond

        # 3. UNet 主体结构
        # 3.1 输入卷积
        x = self.unet.conv_in(sample)
        
        # 3.2 Down Blocks + DEC 注入
        down_block_res_samples = (x,)
        for l, down_block in enumerate(self.unet.down_blocks):
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                x, res_samples = down_block(
                    hidden_states=x, 
                    temb=emb, 
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                x, res_samples = down_block(hidden_states=x, temb=emb)
            
            down_block_res_samples += res_samples

            # === 您的 DEC 注入逻辑 ===
            inj = injections[l]
            if inj.shape[-2:] != x.shape[-2:]:
                inj = F.interpolate(inj, size=x.shape[-2:], mode="bilinear", align_corners=False)
            
            # x = x + inj.to(x.dtype)

            # if l == 0: # 只在第一个注入点和训练迭代初期打印
            #     print("**************************************")
            #     print(f"Level {l}: UNet feat norm: {x.norm().item():.4f}, DEC inj norm: {inj.norm().item():.4f}")
            #     print(f"Level {l}: UNet feat std: {x.std().item():.4f}, DEC inj std: {inj.std().item():.4f}")

            # inj = inj.to(x.dtype)
            # g = self.dec_gates[l].view(1, 1, 1, 1)
            # x = x + g * inj.to(x.dtype)

            # q_in = x.detach().to(x.dtype)
            # y = self.inj_xattn[l](q_in, inj)
            # y = self.inj_norms[l](y).to(x.dtype)
            # x = x + g * self.dec_adapters[l](y)

            q_in = x.to(x.dtype)
            y = self.inj_xattn[l](q_in, inj)
            x = x + y  # atten_4_q

            # ==========================

        # 3.3 Mid Block
        if self.unet.mid_block is not None:
            x = self.unet.mid_block(
                x, 
                emb, 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        
        # 3.4 Up Blocks
        # 检查是否需要处理非标准尺寸
        default_overall_up_factor = 2**self.unet.num_upsamplers
        forward_upsample_size = any(dim % default_overall_up_factor != 0 for dim in sample.shape[-2:])
                
        for i, up_block in enumerate(self.unet.up_blocks):
            is_final_block = i == len(self.unet.up_blocks) - 1
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            upsample_size = None
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                x = up_block(
                    hidden_states=x,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                x = up_block(
                    hidden_states=x, 
                    temb=emb, 
                    res_hidden_states_tuple=res_samples, 
                    upsample_size=upsample_size
                )

        # 4. 输出块 (根据 config 必须包含 norm 和 act)
        # self.unet.conv_norm_out 存在，因为 norm_num_groups = 32
        
        x = self.unet.conv_norm_out(x)
        x = self.unet.conv_act(x)
        # x = UNet2DConditionOutput(x)
            
        x = self.unet.conv_out(x)
        
        return x


    def _unet_with_dec_forward_up(
            self,
            sample: torch.FloatTensor,
            timestep: torch.FloatTensor,
            encoder_hidden_states: torch.Tensor,
            I_ref: torch.FloatTensor = None,
            I_before: torch.FloatTensor = None,
            I_after: torch.FloatTensor = None,
            attention_mask: torch.Tensor = None,
            cross_attention_kwargs: dict = None,
            encoder_attention_mask: torch.Tensor = None,
        ):
        use_dec_injection = self.use_dec and (I_ref is not None and I_before is not None and I_after is not None)

        if not use_dec_injection:
            return self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            ).sample

        # 1. 准备 DEC 注入特征
        with torch.no_grad():
            z_ref    = self.vae.encode(I_ref).latent_dist.mode()    * self.vae.config.scaling_factor
            z_before = self.vae.encode(I_before).latent_dist.mode() * self.vae.config.scaling_factor
            z_after  = self.vae.encode(I_after).latent_dist.mode()  * self.vae.config.scaling_factor
        
        # 反转顺序以匹配 Up Blocks
        # DECNet 输出的 injections 顺序是 [level_0, level_1, level_2, level_3] (从大到小)
        # Up Blocks 的处理顺序是 [level_3, level_2, level_1, level_0] (从小到大)
        injections = self.dec(z_ref, z_before, z_after)
        injections = injections[::-1]  # 反转列表 -> [level_3, level_2, level_1, level_0]

        # 2. 计算时间嵌入 (这部分逻辑保持不变)
        if not torch.is_tensor(timestep):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timestep = torch.tensor([timestep], dtype=dtype, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        
        timesteps = timestep.expand(sample.shape[0])
        
        t_emb = self.unet.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.unet.time_embedding(t_emb)

        # 3. UNet 主体结构
        # 3.1 输入卷积
        x = self.unet.conv_in(sample)
        
        # 3.2 Down Blocks (【修改点 2】: 移除所有注入逻辑)
        down_block_res_samples = (x,)
        for l, down_block in enumerate(self.unet.down_blocks):
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                x, res_samples = down_block(
                    hidden_states=x, 
                    temb=emb, 
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                x, res_samples = down_block(hidden_states=x, temb=emb)
            
            down_block_res_samples += res_samples
            
            # === 您的 DEC 注入逻辑【已移除】 ===
            # 不再在这里进行任何注入
            # =================================

        # 3.3 Mid Block (保持不变)
        if self.unet.mid_block is not None:
            x = self.unet.mid_block(
                x, 
                emb, 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        
        # 3.4 Up Blocks (【修改点 3】: 在这里添加注入逻辑)
        for i, up_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            # --- 原生的 Up Block 前向传播 ---
            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                x = up_block(
                    hidden_states=x,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                x = up_block(
                    hidden_states=x, 
                    temb=emb, 
                    res_hidden_states_tuple=res_samples, 
                )
            
            # --- 【核心修改】在这里执行 DEC 注入 ---
            # i 从 0 到 3，正好对应反转后的 injections 列表
            inj = injections[i]
            
            # 确保注入特征的尺寸与 UNet 特征图一致
            if inj.shape[-2:] != x.shape[-2:]:
                inj = F.interpolate(inj, size=x.shape[-2:], mode="bilinear", align_corners=False)
            
            # 使用互注意力进行注入
            q_in = x.to(x.dtype)
            y = self.inj_xattn[i](q_in, inj) # 注意这里用 i 作为索引
            x = x + y  # atten_4_q
            # -----------------------------------

        # 4. 输出块 (保持不变)
        x = self.unet.conv_norm_out(x)
        x = self.unet.conv_act(x)
        x = self.unet.conv_out(x)
        
        return x

    def _unet_with_dec_forward_check(
            self,
            sample: torch.FloatTensor,
            timestep: torch.FloatTensor,
            encoder_hidden_states: torch.Tensor,
            I_ref: torch.FloatTensor = None,
            I_before: torch.FloatTensor = None,
            I_after: torch.FloatTensor = None,
            attention_mask: torch.Tensor = None,
            cross_attention_kwargs: dict = None,
            encoder_attention_mask: torch.Tensor = None,
        ):
        
        use_dec_injection = self.use_dec and (I_ref is not None and I_before is not None and I_after is not None)

        if not use_dec_injection:
            return self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            ).sample

        # 1. 准备 DEC 注入特征
        with torch.no_grad():
            z_ref    = self.vae.encode(I_ref).latent_dist.mode()    * self.vae.config.scaling_factor
            z_before = self.vae.encode(I_before).latent_dist.mode() * self.vae.config.scaling_factor
            z_after  = self.vae.encode(I_after).latent_dist.mode()  * self.vae.config.scaling_factor
        
        # 【修改点 1】: DECNet 的输出需要反转顺序以匹配 Up Blocks
        # DECNet 输出的 injections 顺序是 [level_0, level_1, level_2, level_3] (从大到小)
        # Up Blocks 的处理顺序是 [level_3, level_2, level_1, level_0] (从小到大)
        injections = self.dec(z_ref, z_before, z_after)
        injections = injections[::-1]  # 反转列表 -> [level_3, level_2, level_1, level_0]

        # 2. 计算时间嵌入 (这部分逻辑保持不变)
        if not torch.is_tensor(timestep):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timestep = torch.tensor([timestep], dtype=dtype, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        
        timesteps = timestep.expand(sample.shape[0])
        
        t_emb = self.unet.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.unet.time_embedding(t_emb)

        # 3. UNet 主体结构
        # 3.1 输入卷积
        x = self.unet.conv_in(sample)
        
        # 3.2 Down Blocks (【修改点 2】: 移除所有注入逻辑)
        down_block_res_samples = (x,)
        for l, down_block in enumerate(self.unet.down_blocks):
            def create_custom_forward(block, *args, **kwargs):
                def custom_forward(*inputs):
                    return block(*inputs, *args, **kwargs)
                return custom_forward
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                x, res_samples = checkpoint(
                    create_custom_forward(down_block, temb=emb, encoder_hidden_states=encoder_hidden_states),
                    x,
                    use_reentrant=False # 推荐使用非重入式，对新版 PyTorch 更友好
                )
            else:
                x, res_samples = checkpoint(
                    create_custom_forward(down_block, temb=emb),
                    x,
                    use_reentrant=False
                )            
            down_block_res_samples += res_samples
            

        # 3.3 Mid Block (保持不变)
        if self.unet.mid_block is not None:
            def create_custom_forward_mid(block, *args, **kwargs):
                def custom_forward(*inputs):
                    return block(*inputs, *args, **kwargs)
                return custom_forward
            x = checkpoint(
                create_custom_forward_mid(self.unet.mid_block, emb, encoder_hidden_states=encoder_hidden_states),
                x,
                use_reentrant=False
            )
        
        # 3.4 Up Blocks + 注入
        for i, up_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            def create_custom_forward_up(block, *args, **kwargs):
                def custom_forward(*inputs):
                    # inputs[0] is hidden_states, inputs[1:] is res_hidden_states_tuple
                    return block(hidden_states=inputs[0], res_hidden_states_tuple=inputs[1:], *args, **kwargs)
                return custom_forward
            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                x = checkpoint(
                    create_custom_forward_up(up_block, temb=emb, encoder_hidden_states=encoder_hidden_states),
                    x, *res_samples,
                    use_reentrant=False
                )
            else:
                x = checkpoint(
                    create_custom_forward_up(up_block, temb=emb),
                    x, *res_samples,
                    use_reentrant=False
                )
            
            # --- 【核心修改】在这里执行 DEC 注入 ---
            # i 从 0 到 3，正好对应反转后的 injections 列表
            inj = injections[i]
            
            # 确保注入特征的尺寸与 UNet 特征图一致
            if inj.shape[-2:] != x.shape[-2:]:
                inj = F.interpolate(inj, size=x.shape[-2:], mode="bilinear", align_corners=False)
            
            # 使用互注意力进行注入
            q_in = x.to(x.dtype)
            y = self.inj_xattn[i](q_in, inj) # 注意这里用 i 作为索引
            x = x + y  # atten_4_q
            # -----------------------------------

        # 4. 输出块 (保持不变)
        x = self.unet.conv_norm_out(x)
        x = self.unet.conv_act(x)
        x = self.unet.conv_out(x)
        
        return x

    def forward(self, x, timesteps=None, prompt=None, prompt_tokens=None,
                I_ref=None, I_before=None, I_after=None):
        """
        x: [B, C, H, W] 像素域（-1~1）
        可选：I_ref/I_before/I_after: [B,3,H,W]（像素域 -1~1），提供则启用 DEC 注入
        """
        assert (timesteps is None) != (self.timesteps is None), "Either timesteps or self.timesteps should be provided"
        caption_enc = self._encode_text(prompt, prompt_tokens)

        z = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor
        Bv = z.shape[0]
        # t = (timesteps if timesteps is not None else self.timesteps).to(z0.device).long() # self.sched.timesteps
        # t = (self.timesteps).to(z.device).long() # self.sched.timesteps
        t = torch.as_tensor(self.timesteps, device=z.device, dtype=torch.long).view(-1)
        if t.ndim == 0 or t.numel() == 1:
            t = t.expand(Bv)                     # [Bv]
        # noise = torch.randn_like(z0)
        # z = self.sched.add_noise(z0, noise, t) # 得到与 t 匹配的噪声等级的输入
        if caption_enc.shape[0] == 1 and Bv > 1:
            caption_enc = caption_enc.expand(Bv, -1, -1) 


        # 调用带注入的 UNet 前向
        model_pred = self._unet_with_dec_forward_up(
            sample=z, timestep=t, encoder_hidden_states=caption_enc,
            I_ref=I_ref, I_before=I_before, I_after=I_after
        )
        # z_denoised = self.sched.step(model_pred, self.timesteps, z, return_dict=True).prev_sample
        out_step = self.sched.step(model_pred, t, z, return_dict=True)  # ← 也用同一个 t
        z0_hat = out_step.prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks

        out = self.vae.decode(z0_hat / self.vae.config.scaling_factor).sample.clamp(-1, 1)
        self.vae.decoder.incoming_skip_acts = None   # 建议加上
        # out = (self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return out


    # def sample(self, image, width, height, ref_image=None, before_image=None, after_image=None,
    #         timesteps=None, prompt=None, prompt_tokens=None):
        
    #     # 1. 保留原始尺寸，用于最后恢复
    #     input_width, input_height = image.size
        
    #     # 2. 【修改】移除这里的 resize 操作
    #     # new_width = image.width - image.width % 8
    #     # new_height = image.height - image.height % 8
    #     # image = image.resize((new_width, new_height), Image.LANCZOS)

    #     T = transforms.Compose([
    #         # 直接使用传入的 width 和 height 进行缩放
    #         transforms.Resize((height, width), interpolation=Image.LANCZOS),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     ])

    #     x = T(image).unsqueeze(0).cuda()

    #     I_ref = I_before = I_after = None
    #     if (ref_image is not None) and (before_image is not None) and (after_image is not None):
    #         # 【修改】同样移除这里的 resize 操作，让 T 来处理
    #         # ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
    #         # before_image = before_image.resize((new_width, new_height), Image.LANCZOS)
    #         # after_image = after_image.resize((new_width, new_height), Image.LANCZOS)
    #         I_ref = T(ref_image).unsqueeze(0).cuda()
    #         I_before = T(before_image).unsqueeze(0).cuda()
    #         I_after = T(after_image).unsqueeze(0).cuda()

    #     output_image = self.forward(
    #         x, timesteps, prompt, prompt_tokens,
    #         I_ref=I_ref, I_before=I_before, I_after=I_after
    #     )
    #     output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        
    #     # 使用原始尺寸恢复输出
    #     output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)
    #     return output_pil

    def sample(self, image, width, height, ref_image=None, before_image=None, after_image=None,
            timesteps=None, prompt=None, prompt_tokens=None):
        
        input_width, input_height = image.size
        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)

        T = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        x = T(image).unsqueeze(0).cuda()

        I_ref = I_before = I_after = None
        if (ref_image is not None) and (before_image is not None) and (after_image is not None):
            # 【修改】同样移除这里的 resize 操作，让 T 来处理
            # ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
            # before_image = before_image.resize((new_width, new_height), Image.LANCZOS)
            # after_image = after_image.resize((new_width, new_height), Image.LANCZOS)
            I_ref = T(ref_image).unsqueeze(0).cuda()
            I_before = T(before_image).unsqueeze(0).cuda()
            I_after = T(after_image).unsqueeze(0).cuda()

        output_image = self.forward(
            x, timesteps, prompt, prompt_tokens,
            I_ref=I_ref, I_before=I_before, I_after=I_after
        )
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        
        # 使用原始尺寸恢复输出
        output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)
        return output_pil



