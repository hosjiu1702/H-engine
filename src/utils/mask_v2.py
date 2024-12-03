from typing import Text
import PIL
import numpy as np
import cv2 as cv
from src.preprocess.humanparsing import Parsing
from src.preprocess.openpose import OpenPose
from src.utils.mask import get_mask_location


class Maskerv2:
    """ Apply Dilated-relaxed Mask Strategy (see section 3.3 in https://arxiv.org/pdf/2411.10499)
    This implementation is heavily based on OOTDiffusion.
    """
    def __init__(self, gpu_id=0):
        self.parser = Parsing(gpu_id)
        self.openpose = OpenPose(gpu_id)
    
    def create_mask(self, img: PIL.Image.Image, category: Text = 'upper_body'):
        keypoints = self.openpose(img)
        body_parse = self.parser(img)
        _mask, _, _, head_mask  = get_mask_location(
            model_type='hd',
            category=category,
            model_parse=body_parse,
            keypoint=keypoints,
            width=img.size[0],
            height=img.size[1]
        )
        mask_np = np.array(_mask)
        contours, _ = cv.findContours(mask_np, cv.RERT_TREE, cv.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv.boundingRect(contours[0])
        mask = cv.rectangle(np.zeros_like(mask_np), (x, y), (x + w, y + h), (255, 255, 255), cv.FILLED)
        redundant_part = np.logical_and(mask, head_mask)
        mask = np.where(redundant_part, mask * 0, mask)

        return mask