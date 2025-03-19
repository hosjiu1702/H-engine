# Credit: AnimateAnyone
# https://github.com/guoqincode/Open-AnimateAnyone/blob/main/models/ReferenceEncoder.py

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()


class ReferenceEncoder(nn.Module):
    def __init__(self, model_path="openai/clip-vit-base-patch32"):
        super(ReferenceEncoder, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(model_path)
        # self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        pooled_output = outputs.pooler_output
        return pooled_output