import torch.cuda
import argparse
# from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from PIL import Image
# from llava.llava_agent import LLavaAgent
# from CKPT_PTH import LLAVA_MODEL_PATH
import os
from torch.nn.functional import interpolate
import time
from tqdm import tqdm

if torch.cuda.device_count() >= 2:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:1'
elif torch.cuda.device_count() == 1:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

# hyparams here
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str)
args = parser.parse_args()
print(args)
TARGET_CAM = {
    'center_camera_fov120': (3840, 2160),
    'center_camera_fov30': (3840, 2160),
    'left_front_camera': (1920, 1080),
    'right_front_camera': (1920, 1080),
    'left_rear_camera': (1920, 1080),
    'right_rear_camera': (1920, 1080),
    'rear_camera': (1920, 1080),
    'front_camera_fov195': (1280, 800),
    'left_camera_fov195': (1280, 800),
    'right_camera_fov195': (1280, 800),
    'rear_camera_fov195': (1280, 800)
}
CAM = ['center_camera_fov120', 'center_camera_fov30', 'left_front_camera', 'right_front_camera',
       'left_rear_camera', 'right_rear_camera', 'rear_camera', 'front_camera_fov195',
       'left_camera_fov195',
       'right_camera_fov195', 'rear_camera_fov195']
auto_collect_dir = os.listdir(args.img_dir)
auto_collect_dir = [dir for dir in auto_collect_dir if os.path.isdir(os.path.join(args.img_dir, dir))
                    and 'AutoCollect' in dir]
index = 0
for auto in auto_collect_dir:  # autocollect
    if auto!='2024_06_20_18_16_06_AutoCollect':
        continue
    p_dir = os.listdir(os.path.join(args.img_dir, auto))
    parse_dir = [dir for dir in p_dir if os.path.isdir(os.path.join(args.img_dir, auto, dir))]  # parse
    for parse in parse_dir:
        path = os.path.join(args.img_dir, auto, parse)
        #if 'rainy' in path or 'evening' in path:
        print(f"Processing {path}")
        os.makedirs(os.path.join(path, 'camera_HQ_delivery'), exist_ok=True)
        for cam in CAM:
            img_HQ_dir = os.path.join(path, 'camera_HQ', cam)
            assert os.path.exists(img_HQ_dir), f"Path {img_HQ_dir} does not exist"
            img_HQ_delivery_dir = os.path.join(path, 'camera_HQ_delivery', cam)
            os.makedirs(img_HQ_delivery_dir, exist_ok=True)
            for img_pth in os.listdir(img_HQ_dir):
                img_name = os.path.splitext(img_pth)[0]  # without .png
                # print(f"Processing {os.path.join(img_dir, img_pth)}")
                HQ_image = Image.open(os.path.join(img_HQ_dir, img_pth))
                # size of HQ image
                HQ_img_size = HQ_image.size
                HQ_image_delivery = HQ_image.resize(TARGET_CAM[cam])
                HQ_image_delivery.save(os.path.join(img_HQ_delivery_dir, img_pth))
