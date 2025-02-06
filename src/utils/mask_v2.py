from typing import Text, Union
import PIL
from PIL import Image
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

    def create_mask(
            self, img: PIL.Image.Image,
            category: Text = 'upper_body',
            return_img: bool = True,
            return_body_parse = False,
            model_type='dc' # default is DressCode
    ) -> Union[np.ndarray, PIL.Image.Image]:
        keypoints, _ = self.openpose(img)
        body_parse, _ = self.parser(img)
        # check the existing of knee keypoints
        right_knee = keypoints['pose_keypoints_2d'][9]
        left_knee = keypoints['pose_keypoints_2d'][12]
        if right_knee == [0, 0] and left_knee == [0, 0]:
            model_type = 'hd'
        _mask, head_mask, _, _  = get_mask_location(
            model_type=model_type,
            category=category,
            model_parse=body_parse,
            keypoint=keypoints,
            width=img.size[0],
            height=img.size[1]
        )
        mask_np = np.array(_mask)
        contours, _ = cv.findContours(mask_np, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv.boundingRect(contours[0])
        mask = cv.rectangle(np.zeros_like(mask_np), (x, y), (x + w, y + h), (255, 255, 255), cv.FILLED)
        redundant_part = np.logical_and(mask, head_mask) # TODO: need to update this logic
        mask = np.where(redundant_part, mask * 0, mask)

        if return_img:
            if return_body_parse:
                return Image.fromarray(mask), body_parse
            return Image.fromarray(mask)

        return mask
