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


def make_sched(timestep=199):
    sched = DDPMScheduler.from_pretrained("/huggingface/sd-turbo", subfolder="scheduler")
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

    if net_difix.dec is None and not any(p.requires_grad for p in net_difix.parameters()):
        print("Warning: DEC is None and no parameters require gradients. Nothing to save.")
        return

    pkg = {
        "meta": {
            "unet_block_out_channels": tuple(net_difix.unet.config.block_out_channels),
        }
    }

    if net_difix.dec is not None:
        pkg["dec_state_dict"] = net_difix.dec.state_dict()

        pkg["meta"]["dec_enc_channels"] = getattr(net_difix.dec, "enc_content", None) and tuple(net_difix.dec.enc_content.out_channels)
        pkg["meta"]["dec_fusion_type"] = type(net_difix.dec.fuses[0]).__name__ if len(net_difix.dec.fuses) > 0 else None

    if net_difix.inj_xattn is not None:
        pkg["inj_xattn_state_dict"] = net_difix.inj_xattn.state_dict()


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
    if "dec_state_dict" in pkg and net_difix.dec is not None:
        net_difix.dec.load_state_dict(pkg["dec_state_dict"])
        print(" -> Loaded 'dec_state_dict'.")
    else:
        print(" -> Warning: 'dec_state_dict' not found in checkpoint or model has no DEC module.")
    if "inj_xattn_state_dict" in pkg and net_difix.inj_xattn is not None:
        net_difix.inj_xattn.load_state_dict(pkg["inj_xattn_state_dict"])
        print(" -> Loaded 'inj_xattn_state_dict'.")
    else:
        print(" -> Warning: 'inj_xattn_state_dict' not found in checkpoint.")
       

    if "meta" in pkg and "unet_block_out_channels" in pkg["meta"]:
        print(f" -> Checkpoint's UNet block out channels: {pkg['meta']['unet_block_out_channels']}")

    if "optimizer" in pkg and optimizer is not None:
        try:
            optimizer.load_state_dict(pkg["optimizer"])
            print(" -> Loaded 'optimizer' state.")
            if new is not None:
                print(f" -> Manually overriding optimizer LR to {new}")
                for param_group in optimizer.param_groups:
                    print(f"last {param_group['lr']}，new{new}")
                    print("*"*50)
                    param_group['lr'] = new
            else:
                for param_group in optimizer.param_groups:
                    print(f"last {param_group['lr']}")
                    print("*"*50)
        except ValueError as e:
            print(f" -> Warning: Could not load optimizer state. It might be from a different model structure. Error: {e}")
    print("Model loading complete.")
    if "lr_scheduler" in pkg and lr_scheduler is not None:
        lr_scheduler.load_state_dict(pkg["lr_scheduler"])
    return net_difix, optimizer, lr_scheduler
    
class InjectXAttnBlock(nn.Module):

    def __init__(self, channels: int, n_heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        assert channels % n_heads == 0, f"channels({channels}) must be divisible by n_heads({n_heads})"
        self.norm_q = nn.GroupNorm(32, channels)
        self.norm_kv = nn.GroupNorm(32, channels)

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=n_heads, batch_first=True)

        self.out_proj = nn.Conv2d(channels, channels, 1)  
        hidden = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x, inj):

        B, C, H, W = x.shape


        q = self.q_proj(self.norm_q(x))          # [B,C,H,W]
        k = self.k_proj(self.norm_kv(inj))       # [B,C,H,W]
        v = self.v_proj(self.norm_kv(inj))       # [B,C,H,W]


        q = q.flatten(2).transpose(1, 2)         # [B, HW, C]
        k = k.flatten(2).transpose(1, 2)         # [B, HW, C]
        v = v.flatten(2).transpose(1, 2)         # [B, HW, C]


        y, _ = self.attn(q, k, v, need_weights=False)  # [B, HW, C]


        y = y.transpose(1, 2).reshape(B, C, H, W)
        y = self.out_proj(y)


        x = x + y


        x = x + self.ffn(x)
        return x
    
class InjectIdentity(nn.Module):

    def __init__(self): super().__init__()
    def forward(self, x, inj): return x


class Difix_my(nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None,
                 ckpt_folder="checkpoints", timestep=999,
                 use_dec=True,
                 dec_enc_channels=(320,640,1280,1280)):
        super().__init__()
        

        self.tokenizer = AutoTokenizer.from_pretrained("/home/yun/workspace/model/huggingface/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("/home/yun/workspace/model/huggingface/sd-turbo", subfolder="text_encoder").cuda()
        self.text_encoder.requires_grad_(False)
        # __init__
        self.timestep = timestep                   
        self.sched = make_sched(self.timestep)             scheduler
        self.timesteps = torch.tensor([self.timestep], device="cuda", dtype=torch.long)


        # VAE
        vae = AutoencoderKL.from_pretrained("/huggingface/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

        unet = UNet2DConditionModel.from_pretrained("/huggingface/sd-turbo", subfolder="unet")



        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            _sd_unet = unet.state_dict()
            for k in sd.get("state_dict_unet", {}):
                if k in _sd_unet: _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            vae.encoder.requires_grad_(False)

        self.unet, self.vae = unet.to("cuda"), vae.to("cuda")
=
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

            for p in self.unet.parameters(): 
                p.requires_grad_(False)
        else:
            self.dec = None  
        

        self.vae.requires_grad_(False)
        def _count(p): return sum(t.numel() for t in p if t.requires_grad)/1e6
        print("="*50)
        print(f"Trainable params — UNet: {_count(self.unet.parameters()):.2f}M")
        print(f"Trainable params — VAE:  {_count(self.vae.parameters()):.2f}M")
        print(f"Trainable params — DEC:  {(_count(self.dec.parameters()) if self.dec else 0.0):.2f}M")

        print("="*50)


    def set_eval(self):
        self.unet.eval();
        self.vae.eval()
        self.unet.requires_grad_(False); 
        self.vae.requires_grad_(False)
        if self.dec is not None: self.dec.eval()

    def set_train(self, mode: bool = True):

        self.unet.eval()
        self.vae.eval()
        if hasattr(self, "text_encoder"):
            self.text_encoder.eval()

        for p in self.unet.parameters(): p.requires_grad_(False)
        for p in self.vae.parameters():  p.requires_grad_(False)
        if hasattr(self, "text_encoder"):
            for p in self.text_encoder.parameters(): p.requires_grad_(False)


        if self.dec is not None:
            self.dec.train()
            for p in self.dec.parameters(): p.requires_grad_(True)


        for blk in self.inj_xattn:
            blk.train()
            for p in blk.parameters(): p.requires_grad_(True)

        
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

        
        with torch.no_grad():
            z_ref    = self.vae.encode(I_ref).latent_dist.mode()    * self.vae.config.scaling_factor
            z_before = self.vae.encode(I_before).latent_dist.mode() * self.vae.config.scaling_factor
            z_after  = self.vae.encode(I_after).latent_dist.mode()  * self.vae.config.scaling_factor
        injections = self.dec(z_ref, z_before, z_after)

        
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
        
       
        timesteps = timestep.expand(sample.shape[0])
        
        
        t_emb = self.unet.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.unet.time_embedding(t_emb) 

        x = self.unet.conv_in(sample)
        

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

 
            inj = injections[l]
            if inj.shape[-2:] != x.shape[-2:]:
                inj = F.interpolate(inj, size=x.shape[-2:], mode="bilinear", align_corners=False)
            
            

            q_in = x.to(x.dtype)
            y = self.inj_xattn[l](q_in, inj)
            x = x + y  # atten_4_q

            
        if self.unet.mid_block is not None:
            x = self.unet.mid_block(
                x, 
                emb, 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        
       
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

        
        
        x = self.unet.conv_norm_out(x)
        x = self.unet.conv_act(x)
        
            
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

       
        with torch.no_grad():
            z_ref    = self.vae.encode(I_ref).latent_dist.mode()    * self.vae.config.scaling_factor
            z_before = self.vae.encode(I_before).latent_dist.mode() * self.vae.config.scaling_factor
            z_after  = self.vae.encode(I_after).latent_dist.mode()  * self.vae.config.scaling_factor
        
        
        injections = self.dec(z_ref, z_before, z_after)
        injections = injections[::-1]  #
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

        
        x = self.unet.conv_in(sample)
        
        
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
            
            
        if self.unet.mid_block is not None:
            x = self.unet.mid_block(
                x, 
                emb, 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        
       
        for i, up_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            
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
            
            
            inj = injections[i]
            
            
            if inj.shape[-2:] != x.shape[-2:]:
                inj = F.interpolate(inj, size=x.shape[-2:], mode="bilinear", align_corners=False)
            
            
            q_in = x.to(x.dtype)
            y = self.inj_xattn[i](q_in, inj) 
            x = x + y  
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

        
        with torch.no_grad():
            z_ref    = self.vae.encode(I_ref).latent_dist.mode()    * self.vae.config.scaling_factor
            z_before = self.vae.encode(I_before).latent_dist.mode() * self.vae.config.scaling_factor
            z_after  = self.vae.encode(I_after).latent_dist.mode()  * self.vae.config.scaling_factor
        
        
        injections = self.dec(z_ref, z_before, z_after)
        injections = injections[::-1]  
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

       
        x = self.unet.conv_in(sample)
        
        
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
                    use_reentrant=False 
                )
            else:
                x, res_samples = checkpoint(
                    create_custom_forward(down_block, temb=emb),
                    x,
                    use_reentrant=False
                )            
            down_block_res_samples += res_samples
            

        
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
        
        
        for i, up_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            def create_custom_forward_up(block, *args, **kwargs):
                def custom_forward(*inputs):
                    
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
            
            
            inj = injections[i]
            
            
            if inj.shape[-2:] != x.shape[-2:]:
                inj = F.interpolate(inj, size=x.shape[-2:], mode="bilinear", align_corners=False)
            
            
            q_in = x.to(x.dtype)
            y = self.inj_xattn[i](q_in, inj) 
            x = x + y  
        x = self.unet.conv_norm_out(x)
        x = self.unet.conv_act(x)
        x = self.unet.conv_out(x)
        
        return x

    def forward(self, x, timesteps=None, prompt=None, prompt_tokens=None,
                I_ref=None, I_before=None, I_after=None):
        
        assert (timesteps is None) != (self.timesteps is None), "Either timesteps or self.timesteps should be provided"
        caption_enc = self._encode_text(prompt, prompt_tokens)

        z = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor
        Bv = z.shape[0]
        
        t = torch.as_tensor(self.timesteps, device=z.device, dtype=torch.long).view(-1)
        if t.ndim == 0 or t.numel() == 1:
            t = t.expand(Bv)                     # [Bv]
       
        if caption_enc.shape[0] == 1 and Bv > 1:
            caption_enc = caption_enc.expand(Bv, -1, -1) 


        
        model_pred = self._unet_with_dec_forward_up(
            sample=z, timestep=t, encoder_hidden_states=caption_enc,
            I_ref=I_ref, I_before=I_before, I_after=I_after
        )
        
        out_step = self.sched.step(model_pred, t, z, return_dict=True)  # ← 也用同一个 t
        z0_hat = out_step.prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks

        out = self.vae.decode(z0_hat / self.vae.config.scaling_factor).sample.clamp(-1, 1)
        self.vae.decoder.incoming_skip_acts = None   
        # out = (self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return out


   

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
            
            I_ref = T(ref_image).unsqueeze(0).cuda()
            I_before = T(before_image).unsqueeze(0).cuda()
            I_after = T(after_image).unsqueeze(0).cuda()

        output_image = self.forward(
            x, timesteps, prompt, prompt_tokens,
            I_ref=I_ref, I_before=I_before, I_after=I_after
        )
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        
        
        output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)
        return output_pil



