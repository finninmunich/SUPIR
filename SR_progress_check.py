import torch.cuda
import argparse
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from PIL import Image
from llava.llava_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH
from data_to_process.selected_bundles.selected_bundles_0927 import SELECTED_BUNDLES_0927
from data_to_process.selected_bundles.selected_bundles_0928 import SELECTED_BUNDLES_0928
from data_to_process.selected_bundles.selected_bundles_1016 import SELECTED_BUNDLES_1016
import os
from torch.nn.functional import interpolate
import time
from tqdm import tqdm
import json
from torch.cuda.amp import autocast, GradScaler
import torch
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
#parser.add_argument("--clip_json", type=str)
parser.add_argument("--input", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--upscale", type=int, default=1)
parser.add_argument("--SUPIR_sign", type=str, default='Q', choices=['F', 'Q'])
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--min_size", type=int, default=1024)
parser.add_argument("--edm_steps", type=int, default=50)
parser.add_argument("--s_stage1", type=int, default=-1)
parser.add_argument("--s_churn", type=int, default=5)
parser.add_argument("--s_noise", type=float, default=1.003)
parser.add_argument("--s_cfg", type=float, default=7.5)
parser.add_argument("--s_stage2", type=float, default=1.)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--a_prompt", type=str,
                    default='Cinematic, High Contrast, highly detailed, taken using a Canon EOS R '
                            'camera, hyper detailed photo - realistic maximum detail, 32k, Color '
                            'Grading, ultra HD, extreme meticulous detailing, skin pore detailing, '
                            'hyper sharpness, perfect without deformations.')
parser.add_argument("--n_prompt", type=str,
                    default='painting, oil painting, illustration, drawing, art, sketch, oil painting, '
                            'cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, '
                            'worst quality, low quality, frames, watermark, signature, jpeg artifacts, '
                            'deformed, lowres, over-smooth')
parser.add_argument("--color_fix_type", type=str, default='Wavelet', choices=["None", "AdaIn", "Wavelet"])
parser.add_argument("--linear_CFG", action='store_true', default=True)
parser.add_argument("--linear_s_stage2", action='store_true', default=False)
parser.add_argument("--spt_linear_CFG", type=float, default=4.0)
parser.add_argument("--spt_linear_s_stage2", type=float, default=0.)
parser.add_argument("--ae_dtype", type=str, default="bf16", choices=['fp32', 'bf16'])
parser.add_argument("--diff_dtype", type=str, default="fp16", choices=['fp32', 'fp16', 'bf16'])
parser.add_argument("--no_llava", action='store_true', default=False)
parser.add_argument("--loading_half_params", action='store_true', default=False)
parser.add_argument("--use_tile_vae", action='store_true', default=False)
parser.add_argument("--encoder_tile_size", type=int, default=512)
parser.add_argument("--decoder_tile_size", type=int, default=64)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
args = parser.parse_args()
print(args)
use_llava = False
CAM = ['center_camera_fov120', 'center_camera_fov30', 'left_front_camera', 'right_front_camera',
       'left_rear_camera', 'right_rear_camera', 'rear_camera', 'front_camera_fov195',
       'left_camera_fov195',
       'right_camera_fov195', 'rear_camera_fov195']
WH_TO_UPSCALE = {
    (3840,2160):2,
    (1920,1080):2,
    (1920,1280):2,
    (1920,1536):2,
    (1536,864):2,
    (1280,800):2
}

# filter bundle that has been processed
processed_list=[]
un_processed_list=[]
input_folder  = args.input
if os.path.exists(os.path.join(input_folder, 'new_trainlist')):
    trainlist_path = os.path.join(input_folder, 'new_trainlist')
    bundle_list = os.listdir(trainlist_path)
    bundle_list = [os.path.join(trainlist_path,trainlist) for trainlist in bundle_list if trainlist.endswith('.json')]
else:
    with open(args.input,'rb') as f:
        bundle_list = [json.loads(line) for line in f]

print(f"Processing {len(bundle_list)} bundles")
SR_BUNDLE = 0
for trainlist_json in tqdm(bundle_list):
    if isinstance(trainlist_json, dict):
        _data = trainlist_json
    else:
        with open(trainlist_json, 'r') as f:
            _data = json.load(f)
    _target_size={}
    valid = True
    for cam in CAM:
        H = _data['camera_infos'][cam]['image_height']
        W = _data['camera_infos'][cam]['image_width']
        _target_size[cam] = (1536,864)
        _filename = _data['camera_infos'][cam]['filename']
        #ac,gp = _filename.split('/camera')[0].split('/')[-2:]
        #assert ac == auto_collect and gp == gtparse, f"Auto collect and gtparse mismatch: {ac} vs {auto_collect}, {gp} vs {gtparse}"
        if 'auto-oss:s3://sdc3-gt-qc-2/pap_finn/' in _filename:
            _file_path = _filename.replace('auto-oss:s3://sdc3-gt-qc-2/pap_finn/','./magicdrive-log/')
        elif "auto-oss:s3://sdc3-gt-qc-2/sdc3_finn/" in _filename:
            _file_path = _filename.replace('auto-oss:s3://sdc3-gt-qc-2/sdc3_finn/','./magicdrive-log/')
        else:
            print(f"Unknown path {_filename}")
            continue
        #print(f"Processing {_file_path}")
        LQ_ips = Image.open(os.path.join(_file_path))
        #print(f"original size: {LQ_ips.size}, target size {_target_size[cam]},upscale by {WH_TO_UPSCALE.get(_target_size[cam],3)}")
        if LQ_ips.size != _target_size[cam]:
            valid = False
            break
    if valid:
        SR_BUNDLE += 1
print(f"Total {SR_BUNDLE} bundles are SR ready, {SR_BUNDLE/len(bundle_list)*100:.2f}%")
