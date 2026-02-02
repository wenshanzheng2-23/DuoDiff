import os
import torch
import json
from glob import glob
from PIL import Image
import torchvision.transforms.functional as F


class PairedInpaintDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=576, width=1024, tokenizer=None):  # height=576, width=1024
        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer



    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        
        input_img = self.data[img_id]["image"]
        before_img = self.data[img_id]["before_image"]
        after_img = self.data[img_id]["after_image"]
        output_img = self.data[img_id]["target_image"]
        ref_img = self.data[img_id]["ref_image"] if "ref_image" in self.data[img_id] else None
        caption = self.data[img_id]["prompt"]
        
        try:
            input_img = Image.open(input_img)
            before_img = Image.open(before_img)
            after_img = Image.open(after_img)
            output_img = Image.open(output_img)
        except:
            print("Error loading image:", input_img, before_img, after_img, output_img)
            return self.__getitem__(idx + 1)


        
        img_t = F.to_tensor(input_img)
        img_t = F.resize(img_t, self.image_size)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        before_t = F.to_tensor(before_img)
        before_t = F.resize(before_t, self.image_size)
        before_t = F.normalize(before_t, mean=[0.5], std=[0.5])

        after_t = F.to_tensor(after_img)
        after_t = F.resize(after_t, self.image_size)
        after_t = F.normalize(after_t, mean=[0.5], std=[0.5])

        output_t = F.to_tensor(output_img)
        output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        if ref_img is not None:
            ref_img = Image.open(ref_img)
            ref_t = F.to_tensor(ref_img)
            ref_t = F.resize(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
        
            # img_t = torch.stack([img_t, ref_t], dim=0)
            # output_t = torch.stack([output_t, ref_t], dim=0)            
        else:
            img_t = img_t.unsqueeze(0)
            output_t = output_t.unsqueeze(0)




        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "ref_pixel_values":ref_t,
            "before_pixel_values":before_t,
            "after_pixel_values":after_t,
            "caption": caption,
        }

        # 如果带 tokenizer，可以用 caption（可选，比如 ref 文件名）
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids
            out["input_ids"] = input_ids

        return out
