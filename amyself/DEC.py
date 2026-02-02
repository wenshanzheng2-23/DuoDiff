import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from diffusers.models.unet_2d_condition import UNet2DConditionOutput


def make_norm(norm: str, num_channels: int):
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        # 常用32组，若不整除退化为1组（等价于IN风格）
        groups = 32 if num_channels % 32 == 0 else max(1, num_channels // 8)
        return nn.GroupNorm(groups, num_channels)
    elif norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "ln":
        # Channel-wise LN（对HW作Instance级，常见替代）
        return nn.GroupNorm(1, num_channels)
    else:
        raise ValueError(f"Unsupported norm: {norm}")

def make_act(act: str):
    act = act.lower()
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "lrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"Unsupported act: {act}")

# ======= 基本卷积块（含残差可选） =======
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm="gn", act="silu", k=3, s=1, p=1, residual=True):
        super().__init__()
        self.residual = residual and (in_ch == out_ch) and (s == 1)
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = make_norm(norm, out_ch)
        self.act = make_act(act)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if hasattr(self.norm, "weight") and self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)
        if hasattr(self.norm, "bias") and self.norm.bias is not None:
            nn.init.zeros_(self.norm.bias)

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        if self.residual:
            y = y + x
        return y

# ======= 抗混叠下采样 =======
class Downsample(nn.Module):
    """
    mode:
      - 'conv'    : 3x3 stride=2 conv
      - 'avgpool' : AvgPool2d(stride=2) + 1x1 conv 调整通道
      - 'blurpool': 先高斯模糊再 stride=2 卷积（轻量抗混叠）
    """
    def __init__(self, in_ch: int, out_ch: int, mode="conv", norm="gn", act="silu"):
        super().__init__()
        self.mode = mode.lower()
        if self.mode == "conv":
            self.op = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                make_norm(norm, out_ch),
                make_act(act),
            )
            nn.init.kaiming_normal_(self.op[0].weight, mode="fan_out", nonlinearity="relu")
        elif self.mode == "avgpool":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.proj = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
            self.norm = make_norm(norm, out_ch)
            self.act = make_act(act)
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
        elif self.mode == "blurpool":
            # 简单 3x3 近似高斯核
            kernel = torch.tensor([[1., 2., 1.],
                                   [2., 4., 2.],
                                   [1., 2., 1.]])
            kernel = kernel / kernel.sum()
            self.register_buffer("blur_kernel", kernel[None, None, :, :])  # [1,1,3,3]
            self.proj = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)
            self.norm = make_norm(norm, out_ch)
            self.act = make_act(act)
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
        else:
            raise ValueError(f"Unsupported downsample mode: {mode}")

    def forward(self, x):
        if self.mode == "conv":
            return self.op(x)
        elif self.mode == "avgpool":
            x = self.pool(x)
            x = self.proj(x)
            x = self.norm(x)
            return self.act(x)
        elif self.mode == "blurpool":
            # 对每个通道做相同的模糊
            B, C, H, W = x.shape
            k = self.blur_kernel.expand(C, 1, 3, 3)  # depthwise
            x = F.conv2d(x, k, bias=None, stride=1, padding=1, groups=C)
            x = self.proj(x)
            x = self.norm(x)
            return self.act(x)

class PyramidEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int,
        channels: List[int],
        norm: str = "gn",
        act: str = "silu",
        ds_mode: str = "conv",
        stage_depth: int = 2,      # 每个 stage 的 ConvBlock 数量（不含下采样）
        residual: bool = True,     # ConvBlock 是否使用残差（in==out 才生效）
    ):
        super().__init__()
        assert len(channels) >= 1, "channels 至少包含一个stage"
        self.out_channels = channels

        # stem：从 in_ch -> channels[0]
        self.stem = ConvBlock(in_ch, channels[0], norm=norm, act=act, residual=False)

        blocks = []
        stage_indices = []  # 记录每个 stage 结束时的下标
        c_list = channels

        for i in range(len(c_list)):
            Cin = c_list[i]
            # stage 内堆叠若干 ConvBlock（保持通道不变）
            for _ in range(stage_depth):
                blocks.append(ConvBlock(Cin, Cin, norm=norm, act=act, residual=residual))
            # 记录：该 stage 的最后一个 ConvBlock 的索引
            stage_indices.append(len(blocks) - 1)

            # 下采样到下一 stage
            if i < len(c_list) - 1:
                blocks.append(Downsample(c_list[i], c_list[i + 1], mode=ds_mode, norm=norm, act=act))

        self.body = nn.ModuleList(blocks)
        self.stage_indices = tuple(stage_indices)  # 不可变，便于下游使用

    @torch.no_grad()
    def infer_feat_shapes(self, H: int, W: int) -> List[Tuple[int, int, int]]:
        """给定输入分辨率，推断每个 stage 输出的 (C,H,W)（不跑真实前向）。"""
        shapes = []
        h, w = H, W
        for i, C in enumerate(self.out_channels):
            shapes.append((C, h, w))
            if i < len(self.out_channels) - 1:
                h, w = (h + 1) // 2, (w + 1) // 2  # 近似（与stride=2一致）
        return shapes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feats = []
        x = self.stem(x)  # [B, C0, H, W]
        for li, layer in enumerate(self.body):
            x = layer(x)
            # 若该层是某个 stage 的最后一个 ConvBlock，则收集特征
            if li in self.stage_indices:
                feats.append(x)
        return tuple(feats)
    

class ZeroConv(nn.Module):
    """
    1x1 投射到 UNet 对应通道；init:
      - "zero": 完全零（经典 ZeroConv）
      - "small": N(0, std^2) 小随机初始化
    """
    def __init__(self, in_ch: int, out_ch: int, init: str = "small", std: float = 1e-3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.reset_parameters(init, std)

    def reset_parameters(self, init: str, std: float):
        if init == "zero":
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        elif init == "small":
            nn.init.normal_(self.proj.weight, mean=0.0, std=std)
            nn.init.zeros_(self.proj.bias)
        else:
            raise ValueError(f"Unknown proj init: {init}")

    def forward(self, x):
        return self.proj(x)


# ====== 2) ConcatFuse1x1：concat -> 1x1 -> norm+act -> (可选1层3x3增强) ======
class ConcatFuse1x1(nn.Module):
    def __init__(self, c_content: int, c_struct: int, c_out: int,
                 norm="gn", act="silu", enhance: bool = False):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(c_content + c_struct, c_out, kernel_size=1, bias=False),
            make_norm(norm, c_out),
            make_act(act),
        )
        self.enhance = enhance
        if enhance:
            self.post = ConvBlock(c_out, c_out, norm=norm, act=act, residual=True)

        # 初始化
        nn.init.kaiming_normal_(self.fuse[0].weight, mode="fan_out", nonlinearity="relu")

    def forward(self, fc, fs):
        x = torch.cat([fc, fs], dim=1)
        x = self.fuse(x)
        if self.enhance:
            x = self.post(x)
        return x


# ====== 3) CrossAttentionFuse：Q=content, K/V=struct，多头注意力 + 残差门控 ======
class CrossAttentionFuse(nn.Module):
    """
    把结构特征注入内容：Attn(fc <- fs)
    - 采用 1x1 线性生成 Q,K,V；空间展平到 HW 维做注意力；最后投影回 C 并加门控残差
    - 更稳的训练：输出残差门控参数 alpha 初始为 0
    """
    def __init__(self, c_content: int, c_struct: int, c_out: int,
                 n_heads: int = 8, norm="gn", act="silu"):
        super().__init__()
        assert c_out % n_heads == 0, "c_out 必须能被 n_heads 整除"
        self.c_out = c_out
        self.n_heads = n_heads
        self.head_dim = c_out // n_heads

        self.q_proj = nn.Conv2d(c_content, c_out, 1, bias=False)
        self.k_proj = nn.Conv2d(c_struct,  c_out, 1, bias=False)
        self.v_proj = nn.Conv2d(c_struct,  c_out, 1, bias=False)
        self.out_proj = nn.Conv2d(c_out,   c_out, 1, bias=False)
        self.norm = make_norm(norm, c_out)
        self.act  = make_act(act)
        if c_content == c_out:
            self.res_proj = nn.Identity()
        else:
            self.res_proj = nn.Conv2d(c_content, c_out, kernel_size=1, bias=False)

        # 残差门控，初始为0，等价于“从0开始逐步引入注意力”
        self.alpha = nn.Parameter(torch.zeros(1))

        # 初始化
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # norm 参数在 make_norm 内置初始化
        if isinstance(self.res_proj, nn.Conv2d):
            nn.init.kaiming_normal_(self.res_proj.weight, mode="fan_in", nonlinearity='linear')

    def forward(self, fc, fs):
        B, Cc, H, W = fc.shape
        _, Cs, _, _ = fs.shape

        q = self.q_proj(fc)  # [B, C, H, W]
        k = self.k_proj(fs)
        v = self.v_proj(fs)

        # 变形为多头：[B, heads, head_dim, HW]
        def reshape_heads(x):
            B, C, H, W = x.shape
            x = x.view(B, self.n_heads, self.head_dim, H * W)
            return x

        q = reshape_heads(q)   # [B, h, d, N]
        k = reshape_heads(k)   # [B, h, d, N]
        v = reshape_heads(v)   # [B, h, d, N]

        # 注意力：q^T k / sqrt(d) -> softmax -> 与 v 相乘
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) / (self.head_dim ** 0.5)  # [B,h,N,N]
        attn = attn.softmax(dim=-1)
        y = torch.einsum('bhij,bhdj->bhdi', attn, v)  # [B,h,d,N]

        # y = F.scaled_dot_product_attention(q, k, v) # [B, h, d, N]

        # 合并头 -> [B, C, H, W]
        y = y.contiguous().view(B, self.c_out, H, W)
        y = self.out_proj(y)
        y = self.norm(y)
        y = self.act(y)

        fc_residual = self.res_proj(fc)

        out = fc_residual + self.alpha * y
        return out


# ====== 4) DECNet 主体 ======
class DECNet(nn.Module):
    """
    双编码器 + 每层融合 + ZeroConv 投射到 UNet 对应通道
    - 输入为 VAE 潜空间：I_ref: [B,4,H,W]；结构为 [I_before, I_after]: [B,8,H,W]
    - 输出 injections: 与 UNet block_out_channels 对齐的多尺度注入
    """
    def __init__(
        self,
        enc_channels: Tuple[int, ...]           = (320, 640, 1280, 1280),
        unet_block_out_channels: Tuple[int, ...]= (320, 640, 1280, 1280),
        fusion_type: str                        = "cross_attn",   # "concat_1x1" | "cross_attn"
        n_heads_cross: int                      = 8,
        norm: str                               = "gn",
        act: str                                = "silu",
        ds_mode: str                            = "blurpool",
        stage_depth: int                        = 3,              # 传递给 PyramidEncoder
        residual: bool                          = True,           # 传递给 PyramidEncoder
        proj_init: str                          = "small",        # "small" | "zero"
        proj_std: float                         = 1e-3,
        interaction_type                        = None ,   # "transformer"
    ):
        super().__init__()
        assert len(enc_channels) == len(unet_block_out_channels), "通道层数需与UNet对齐"
        self.levels = len(enc_channels)

        # === 输入通道（潜空间）===
        in_ch_content, in_ch_struct = 4, 8
        self.interaction_type = interaction_type
        # === 1. 定义结构交互模块 ===
        if self.interaction_type == "3d_conv":
            in_ch_struct = 4
            # kernel_size=(2, 3, 3) 表示在时间维度上跨越2帧，空间上用3x3
            self.struct_interaction = nn.Conv3d(
                in_channels=in_ch_struct,       # 输入特征图的通道数
                out_channels=in_ch_struct,      # 输出通道数保持不变
                kernel_size=(2, 3, 3),  # (T, H, W)
                padding=(0, 1, 1)       # (T, H, W)，时间上不补丁，空间上补丁
            )
        elif self.interaction_type == "transformer":
            in_ch_struct = 4
            # 定义一个 Transformer Encoder 层
            # d_model 是 token 的维度，这里等于通道数 c_in
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=in_ch_struct,         
                nhead=4,        # 可以复用 n_heads 参数
                dim_feedforward=in_ch_struct * 4, # 常规设置
                dropout=0.1,
                activation="gelu",
                batch_first=True      # 重要！我们的输入是 (B, N, D)
            )
            # 可以堆叠多层以增强建模能力，这里用一层作为示例
            self.struct_interaction = nn.TransformerEncoder(transformer_layer, num_layers=1)
            # 位置编码是可选但推荐的，这里简化暂不添加，但可以加
            # self.pos_embed = nn.Parameter(torch.zeros(1, H*W*2, c_in)) 
            # self.pos_embed_before = nn.Parameter(torch.zeros(1, H*W, in_ch_struct))
            # self.pos_embed_after  = nn.Parameter(torch.zeros(1, H*W, in_ch_struct))
        # === 两个金字塔编码器 ===
        self.enc_content = PyramidEncoder(
            in_ch=in_ch_content,
            channels=list(enc_channels),
            norm=norm, act=act, ds_mode=ds_mode,
            stage_depth=stage_depth, residual=residual,
        )
        self.enc_struct  = PyramidEncoder(
            in_ch=in_ch_struct,
            channels=list(enc_channels),
            norm=norm, act=act, ds_mode=ds_mode,
            stage_depth=stage_depth, residual=residual,
        )

        # === 每层融合 + 投射到 UNet 对应通道 ===
        fuses, zeros = [], []
        for l in range(self.levels):
            c = enc_channels[l]
            u = unet_block_out_channels[l]

            if fusion_type == "concat_1x1":
                fuses.append(ConcatFuse1x1(c, c, c, norm=norm, act=act, enhance=False))
            elif fusion_type == "cross_attn":
                fuses.append(CrossAttentionFuse(c, c, c, n_heads=n_heads_cross, norm=norm, act=act))
            else:
                raise ValueError(f"Unknown fusion_type: {fusion_type}")

            zeros.append(ZeroConv(c, u, init=proj_init, std=proj_std))

        self.fuses = nn.ModuleList(fuses)
        self.zeros = nn.ModuleList(zeros)
        # init_val = 1e-2 
        # self.injection_scales = nn.Parameter(torch.full((self.levels,), init_val))


    @staticmethod
    def _align(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """空间尺寸不一致时雙線性对齐。"""
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return a, b

    def forward(self, I_ref: torch.Tensor, I_before: torch.Tensor, I_after: torch.Tensor):
        """
        I_ref    : [B,4,H,W] (VAE latent)
        I_before : [B,4,H,W]
        I_after  : [B,4,H,W]
        return   : list[ level -> [B, u_l, h_l, w_l] ]
        """
        B, C, H, W = I_before.shape
        if self.interaction_type == "transformer":
            before_flat = I_before.flatten(2).permute(0, 2, 1)
            after_flat = I_after.flatten(2).permute(0, 2, 1) 
            # before_flat = I_before.flatten(2).permute(0, 2, 1) + self.pos_embed_before
            # after_flat = I_after.flatten(2).permute(0, 2, 1) + self.pos_embed_after
            struct_tokens = torch.cat([before_flat, after_flat], dim=1)
            interacted_tokens = self.struct_interaction(struct_tokens)
            #    或者可以将两部分结果平均，这里取前一半更简单
            #    [B, 2*H*W, C] -> [B, H*W, C]
            interacted_before = interacted_tokens[:, :H*W, :]  # [B, H*W, C]
            interacted_after  = interacted_tokens[:, H*W:, :]  # [B, H*W, C]
            # 将它们取平均
            output_tokens = (interacted_before + interacted_after) / 2.0   
            interacted_struct = output_tokens.permute(0, 2, 1).view(B, C, H, W)
        elif self.interaction_type == "3d_conv":
            struct_input_3d = torch.stack([I_before, I_after], dim=2)
            interacted_struct_3d = self.struct_interaction(struct_input_3d)
            interacted_struct = interacted_struct_3d.squeeze(2)# c. 移除多余的 "时间" 维度，变回 2D 特征图  [B, C, 1, H, W] -> [B, C, H, W]
        else: # 原始的 concat 方式
            interacted_struct = torch.cat([I_before, I_after], 1)    
        
        
        f_content = self.enc_content(I_ref)                          # tuple per level
        # f_struct  = self.enc_struct(torch.cat([I_before, I_after], 1))
        f_struct = self.enc_struct(interacted_struct) 

        injections = []
        for l in range(self.levels):
            fc, fs = f_content[l], f_struct[l]
            fc, fs = self._align(fc, fs)
            fused  = self.fuses[l](fc, fs)        # [B, c_l, h, w]
            inj    = self.zeros[l](fused)         # [B, u_l, h, w]
            injections.append(inj)
            # scaled_inj = inj * self.injection_scales[l]
            # injections.append(scaled_inj)

        return injections
# 说明与默认值
# 输入空间：这版以 潜空间 为准（4ch/图），如果你要用 RGB（3ch/图），把 in_ch_content=3, in_ch_struct=6 并在外部加一个 VAE 编码器即可。

# 下采样：默认 ds_mode="blurpool" 以减少 aliasing，用于 NVS/重建更稳；追求速度可切 conv。

# 归一化：默认 GroupNorm，小 batch 稳定；你原来的 bn 也可用。

# 融合：concat_1x1 轻量稳定；cross_attn 更强表达，带 alpha 门控防训练早期震荡。

# 投射初始化：

# proj_init="zero"：经典 ZeroConv，初始不扰动 UNet；

# proj_init="small"：小随机（默认 std=1e-3），可避免“全零梯度路径”。

# 用法示例
# enc_channels = (320,640,1280,1280)
# unet_out     = (320,640,1280,1280)

# net = DECNet(enc_channels=enc_channels,
#              unet_block_out_channels=unet_out,
#              fusion_type="concat_1x1",
#              norm="gn", act="silu",
#              ds_mode="blurpool",
#              stage_depth=2,
#              proj_init="small", proj_std=1e-3).cuda()

# B,H,W = 2,256,256
# z_ref    = torch.randn(B,4,H,W, device="cuda")
# z_before = torch.randn(B,4,H,W, device="cuda")
# z_after  = torch.randn(B,4,H,W, device="cuda")

# injections = net(z_ref, z_before, z_after)
# # injections[l]: [B, unet_block_out_channels[l], H_l, W_l]



