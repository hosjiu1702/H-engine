import torch
from diffusers import UNet2DConditionModel


class PriorModelEvolution:

    def __init__(self, device='cuda'):
        self.shaper = UNet2DConditionModel.from_pretrained(
            'Lykon/dreamshaper-8',
            subfolder='unet',
            torch_dtype=torch.float16
        ).to(device)
        self.base = UNet2DConditionModel.from_pretrained(
            'stable-diffusion-v1-5/stable-diffusion-v1-5',
            subfolder='unet',
            torch_dtype=torch.float16
        ).to(device)

    def __call__(self, inpaint: UNet2DConditionModel):
        """
        Prior Model Evolution
        (section 3.3 in https://arxiv.org/abs/2405.18172).
        This implementation use in-place PyTorch operation for tensors.

        Args:
            inpaint: Inpainting unet version

        Returns:
            Evolved Unet
        """
        # default values from paper
        alpha = 1.0
        beta = 1.1

        # alpha * (W_inp - W_base)
        for (name, param1), (_, param2) in zip(inpaint.named_parameters(), self.base.named_parameters()):
            if 'conv_in' in name:
                with torch.no_grad():
                    if len(param1.data.shape) == 4:
                        # conv weights
                        param1.data[:, :4, :, :].sub_(param2) * alpha
                        param1.data[:, 4:, :, :] * alpha
                    else:
                        # conv bias
                        param1.data.sub_(param2) * alpha
            else:
                with torch.no_grad():
                    param1.data.sub_(param2) * alpha

        # beta * (W_DS - W_base)
        for (name, param1), (_, param2) in zip(self.shaper.named_parameters(), self.base.named_parameters()):
            with torch.no_grad():
                param1.data.sub_(param2) * beta
        
        # W_base + alpha * (W_inp - W_base) + beta * (W_DS - W_base)
        # Here, we treat W_inp as W_base in the above expression to implement easier.
        for (name, param1), (_, param2), (_, param3) in zip(
                inpaint.named_parameters(),
                self.base.named_parameters(),
                self.shaper.named_parameters()
        ):
            if 'conv_in' in name:
                if len(param1.data.shape) == 4:
                    param1.data[:, :4, :, :].add_(param2)
                    param1.data[:, :4, :, :].add_(param3)
                else:
                    param1.data.add_(param2)
                    param1.data.add_(param3)
            else:
                param1.data.add_(param2)
                param1.data.add_(param3)
        
        return inpaint