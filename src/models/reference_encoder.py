# Credit: AnimateAnyone, IP-Adapter
# https://github.com/guoqincode/Open-AnimateAnyone/blob/main/models/ReferenceEncoder.py

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers import logging
from src.models.resampler import Resampler
logging.set_verbosity_warning()
logging.set_verbosity_error()


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ReferenceEncoder(nn.Module):
    def __init__(self, huggingface_id="openai/clip-vit-base-patch32", enable_resampler: bool = False, enable_mlp: bool = False, **kwargs):
        super(ReferenceEncoder, self).__init__()
        self.enable_resampler = enable_resampler
        self.enable_mlp = enable_mlp
        
        if not self.enable_resampler and not self.enable_mlp:
            self.model = CLIPVisionModel.from_pretrained(huggingface_id)
        else:
            self.model = CLIPVisionModelWithProjection.from_pretrained(huggingface_id)
            if self.enable_resampler:
                self.image_proj_model = Resampler(
                    dim=kwargs['unet'].config.cross_attention_dim,
                    depth=4,
                    dim_head=64,
                    heads=12,
                    num_queries=16,
                    embedding_dim=self.model.config.hidden_size,
                    output_dim=kwargs['unet'].config.cross_attention_dim,
                    ff_mult=4
                )
            elif self.enable_mlp:
                self.mlp = MLP(self.model.config.hidden_size)

    def forward(self, pixel_values: torch.Tensor):
        image_embeds = self.model(pixel_values, output_hidden_states=True).hidden_states[-2]

        if self.enable_resampler:
            output = self.image_proj_model(image_embeds)
            return output
        elif self.enable_mlp:
            output = self.mlp(image_embeds)
            return output
        else:
            output = self.model(pixel_values)
            return output.pooler_output.squeeze(1)
