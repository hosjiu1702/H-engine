import pdb
from pathlib import Path
import sys
import os
import torch
import onnxruntime as ort
from src.preprocess.humanparsing.parsing_api import onnx_inference
from src.utils import get_project_root


PROJECT_ROOT_PATH = get_project_root()


class Parsing:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.add_session_config_entry('gpu_id', str(gpu_id))
        self.session = ort.InferenceSession(os.path.join(PROJECT_ROOT_PATH, 'checkpoints/humanparsing/parsing_atr.onnx'),
                                            sess_options=session_options,
                                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.lip_session = ort.InferenceSession(os.path.join(PROJECT_ROOT_PATH, 'checkpoints/humanparsing/parsing_lip.onnx'),
                                                sess_options=session_options,
                                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        

    def __call__(self, input_image):
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        return parsed_image, face_mask
