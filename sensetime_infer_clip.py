import argparse
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from PIL import Image
import os
import time
from tqdm import tqdm
import json
from torch.cuda.amp import autocast, GradScaler
import torch
from aoss_client.client import Client
CEPH_CLIENT = Client('aoss.conf')
JISHU_PREFIX='devsftfj:s3://iag-finn-data-generation/data/pap/',
ZIYAN_PREFIX='devsftfj:s3://iag-finn-data-generation/data/ziyan/'
PRODUCTION_DIR = 'fishbone_urban_t68_dusk_1231'
def get_aoss_path(filename):
    assert PRODUCTION_DIR in filename, f"production dir {PRODUCTION_DIR} not in filename {filename}"
    if f'auto-oss:s3://sdc3-gt-qc-2/pap_finn/{PRODUCTION_DIR}/' in filename:
        return filename.replace(f'auto-oss:s3://sdc3-gt-qc-2/{production_dir}/pap/',JISHU_PREFIX)
    elif f"auto-oss:s3://sdc3-gt-qc-2/sdc3_finn/{PRODUCTION_DIR}/" in filename:
        return filename.replace(f"auto-oss:s3://sdc3-gt-qc-2/sdc3_finn/{PRODUCTION_DIR}/",ZIYAN_PREFIX)
    else:
        print(f"Unknown path {filename}")
        return None
def _load_image(rgb_file):
    if "s3://" in rgb_file:
        img_bytes = self.ceph_mclient.get(rgb_file)
        try:
            image = Image.open(io.BytesIO(img_bytes))
        except:
            print(rgb_file)
    else:
        image = Image.open(rgb_file)
    return image

def _load_json(path):
    if "s3://" in path:
        return json.loads(
            self.ceph_mclient.get(path))
    else:
        with open(path) as fp:
            return json.load(fp)



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
parser.add_argument("--process", type=int,default=0)
parser.add_argument("--num_split", type=int,default=8)
parser.add_argument("--input", type=str, default=None)
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

input_folder  = args.input

if os.path.exists(os.path.join(input_folder, 'new_trainlist')):
    trainlist_path = os.path.join(input_folder, 'new_trainlist')
    bundle_list = os.listdir(trainlist_path)
    bundle_list = [os.path.join(trainlist_path,trainlist) for trainlist in bundle_list if trainlist.endswith('.json')]
else:
    with open(args.input,'rb') as f:
        bundle_list = [json.loads(line) for line in f]

subset = len(bundle_list) // int(args.num_split)
if args.process==-1:
    bundle_list = bundle_list
elif args.process == int(args.num_split) - 1:
    bundle_list = bundle_list[args.process * subset:]
else:
    bundle_list = bundle_list[args.process * subset:(args.process + 1) * subset]
print(f"we totally have {len(bundle_list)} bundles to process")

#load SUPIR

model = create_SUPIR_model('options/SUPIR_v0.yaml', SUPIR_sign=args.SUPIR_sign)
if args.loading_half_params:
    model = model.half()
if args.use_tile_vae:
    model.init_tile_vae(encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size)
model.ae_dtype = convert_dtype(args.ae_dtype)
model.model.dtype = convert_dtype(args.diff_dtype)
model = model.to(SUPIR_device)


# processing images

for trainlist_json in tqdm(bundle_list):
    # get trainlist json
    if isinstance(trainlist_json, dict):
        _data = trainlist_json
    else:
        with open(trainlist_json, 'r') as f:
            _data = json.load(f)
    _target_size={}
    start_time = time.time()
    for cam in CAM:
        H = _data['camera_infos'][cam]['image_height']
        W = _data['camera_infos'][cam]['image_width']
        _target_size[cam] = (1536,864)
        _filename = _data['camera_infos'][cam]['filename']
        if 'auto-oss:s3://sdc3-gt-qc-2/pap_finn/' in _filename:
            _file_path = _filename.replace('auto-oss:s3://sdc3-gt-qc-2/pap_finn/','./magicdrive-log/')
        elif "auto-oss:s3://sdc3-gt-qc-2/sdc3_finn/" in _filename:
            _file_path = _filename.replace('auto-oss:s3://sdc3-gt-qc-2/sdc3_finn/','./magicdrive-log/')
        else:
            print(f"Unknown path {_filename}")
            continue
        print(f"Processing {_file_path}")
        LQ_ips = Image.open(os.path.join(_file_path))
        print(f"original size: {LQ_ips.size}, target size {_target_size[cam]},upscale by {WH_TO_UPSCALE.get(_target_size[cam],3)}")
        if LQ_ips.size == _target_size[cam]:
            print(f"Already in target size, skipping")
            continue
        LQ_img, h0, w0 = PIL2Tensor(LQ_ips, upsacle=WH_TO_UPSCALE.get(_target_size[cam],2), min_size=args.min_size)
        LQ_img = LQ_img.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

        captions = ['']
        # print(captions)
        print(f"Processing {cam}")
        # # step 3: Diffusion Process
        with torch.no_grad():
            with autocast():
                samples = model.batchify_sample(LQ_img, captions, num_steps=args.edm_steps, restoration_scale=args.s_stage1,
                                                s_churn=args.s_churn,
                                                s_noise=args.s_noise, cfg_scale=args.s_cfg, control_scale=args.s_stage2,
                                                seed=args.seed,
                                                num_samples=args.num_samples, p_p=args.a_prompt, n_p=args.n_prompt,
                                                color_fix_type=args.color_fix_type,
                                                use_linear_CFG=args.linear_CFG,
                                                use_linear_control_scale=args.linear_s_stage2,
                                                cfg_scale_start=args.spt_linear_CFG,
                                                control_scale_start=args.spt_linear_s_stage2)
        # save
        for _i, sample in enumerate(samples):
            output_img = Tensor2PIL(sample, h0, w0)
            output_img = output_img.resize( _target_size[cam])
            output_img.save(_file_path)
        print(f"Saved {cam} at {_file_path}")
    end_time = time.time()
    print(f"Time taken for this bundle: {end_time - start_time} s ")