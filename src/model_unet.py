import os
import requests
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler
from peft import LoraConfig
from einops import rearrange, repeat


def make_sched(timestep=999):
    noise_scheduler = DDPMScheduler.from_pretrained(
        "/home/yun/workspace/model/huggingface/sd-turbo", subfolder="scheduler"
    )
    noise_scheduler.set_timesteps(1, device="cuda")  # 只走一步
    noise_scheduler.timesteps = torch.tensor([timestep], device="cuda")  # 强制改成 [timestep]
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.cuda()
    return noise_scheduler


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
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)

    # 直接走 up_blocks，不再加 skip
    for up_block in self.up_blocks:
        sample = up_block(sample, latent_embeds)

    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def load_ckpt_from_state_dict(net_difix, optimizer, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cpu")

    if "state_dict_unet" in sd:
        _sd_unet = net_difix.unet.state_dict()
        for k in sd["state_dict_unet"]:
            if k in _sd_unet:
                _sd_unet[k] = sd["state_dict_unet"][k]
        net_difix.unet.load_state_dict(_sd_unet)

    if "optimizer" in sd:
        optimizer.load_state_dict(sd["optimizer"])

    return net_difix, optimizer


def save_ckpt(net_difix, optimizer, outf):
    sd = {}
    sd["state_dict_unet"] = net_difix.unet.state_dict()
    sd["optimizer"] = optimizer.state_dict()
    torch.save(sd, outf)


class Difix(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None,
                 ckpt_folder="checkpoints", lora_rank_vae=4,
                 lora_rank_unet=8, mv_unet=False, timestep=999):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("/home/yun/workspace/model/huggingface/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("/home/yun/workspace/model/huggingface/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_sched(timestep)

        vae = AutoencoderKL.from_pretrained("/home/yun/workspace/model/huggingface/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

        if mv_unet:
            from mv_unet import UNet2DConditionModel
        else:
            from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("/home/yun/workspace/model/huggingface/sd-turbo", subfolder="unet")

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            vae.encoder.requires_grad_(False)

        unet.to("cuda")
        vae.to("cuda")

        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([timestep], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        print("=" * 50)
        print(f"Number of trainable parameters in UNet: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in VAE: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        print("=" * 50)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self, mode=None):
        self.unet.train()
        self.vae.train()
        self.unet.requires_grad_(True)
        self.vae.requires_grad_(False)

    def forward(self, x, timesteps=None, prompt=None, prompt_tokens=None):
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"
        assert (timesteps is None) != (self.timesteps is None), "Either timesteps or self.timesteps should be provided"

        if prompt is not None:
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True,
                                            return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        num_views = x.shape[1]
        x = rearrange(x, 'b v c h w -> (b v) c h w')
        z = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
        caption_enc = repeat(caption_enc, 'b n c -> (b v) n c', v=num_views)

        unet_input = z
        model_pred = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc).sample
        z_denoised = self.sched.step(model_pred, self.timesteps, z, return_dict=True).prev_sample

        output_image = (self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=num_views)

        return output_image
    
    def sample(self, image, width, height, ref_image=None, timesteps=None, prompt=None, prompt_tokens=None):
        input_width, input_height = image.size
        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        T = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        if ref_image is None:
            x = T(image).unsqueeze(0).unsqueeze(0).cuda()
        else:
            ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
            x = torch.stack([T(image), T(ref_image)], dim=0).unsqueeze(0).cuda()
        
        output_image = self.forward(x, timesteps, prompt, prompt_tokens)[:, 0]
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)
        
        return output_pil

    def save_model(self, outf, optimizer):
        sd = {}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["optimizer"] = optimizer.state_dict()
        torch.save(sd, outf)
