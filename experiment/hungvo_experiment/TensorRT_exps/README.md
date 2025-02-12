# TensorRT 

<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://sjbfwnygg89.sg.larksuite.com/wiki/Qq8xwqC5yivuWJkIF81lK4Fig0c" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Document-View%20Report-blue?style=flat&logo=microsoft-word&logoColor=blue' alt='View Report'>
  </a>
</div>


## Reference Link


- [A tutorial about how to build a TensorRT Engine from a PyTorch Model with the help of ONNX](https://github.com/RizhaoCai/PyTorch_ONNX_TensorRT)
- [Theory of TensorRT](https://viblo.asia/p/tensorrt-su-vuot-troi-voi-bai-toan-toi-uu-mo-hinh-deep-learning-y3RL1AayLao)

This experiment aims to speed up stable diffusion using TensorRT



## Download Resources 
### IP Adapter  
Download the IP adapter using the following command:  
```bash
cd models
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin
```

## Inference
### 1. SD1.5 Inpainting + Ip_adapter:
```
python test_sd_inpaint.py
```


### 2. SD1.5 Inpainting + Ip_adapter with TensorRT:
```
  python test_sd_tensorrt_inpaint.py
```
### 3. SD1.5 Inpainting + Ip_adapter + Openpose_controlnet with TensorRT:
```
  python test_sd_tensorrt_inpaint_inpaint_controlnet.py
```


## TODO List
- [x] TensorRT with IP_adapter
- [x] TensorRT with IP_adapter + Controlnet
- []  TensorRT with IP_adapter_plus + Controlnet
- []  NVIDIA Triton Inference Server
