import argparse
import torch
import torch.multiprocessing as mp
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from PIL import Image
import os
import json
from tqdm import tqdm
import logging

CAM = ['center_camera_fov120', 'center_camera_fov30', 'left_front_camera', 'right_front_camera',
       'left_rear_camera', 'right_rear_camera', 'rear_camera', 'front_camera_fov195',
       'left_camera_fov195',
       'right_camera_fov195', 'rear_camera_fov195']
WH_TO_UPSCALE = {
    (3840, 2160): 2,
    (1920, 1080): 2,
    (1920, 1280): 2,
    (1920, 1536): 2,
    (1536, 864): 2,
    (1280, 800): 2
}
class TqdmToLogger:
    """
    将 tqdm 的输出重定向到 logger。
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        # 移除多余的换行符，并通过 logger 输出
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        # tqdm 需要 flush，但在这里不需要实际操作
        pass
def setup_logger(gpu_id,log_dir = "./logs"):
    """为每个 GPU 配置独立的日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"gpu_{gpu_id}.log")

    logger = logging.getLogger(f"GPU_{gpu_id}")
    logger.setLevel(logging.DEBUG)

    # 移除旧的 Handler，避免重复记录日志
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件日志记录
    file_handler = logging.FileHandler(log_file,mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 控制台日志记录（可选）
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(f"[GPU {gpu_id}] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
def process_image(image_path, target_size, model, device,logger,args):
    """处理单张图像并保存"""
    try:
        LQ_img = Image.open(image_path)
        if LQ_img.size == target_size:
            logger.info(f"Image {image_path} already at target size, skipping.")
            return
        logger.info(f"original size: {LQ_img.size}, target size {target_size},upscale by {WH_TO_UPSCALE.get(target_size,2)}")
        LQ_tensor, h0, w0 = PIL2Tensor(LQ_img, upsacle=WH_TO_UPSCALE.get(target_size, 2), min_size=args.min_size)
        LQ_tensor = LQ_tensor.unsqueeze(0).to(device)[:, :3, :, :]
        captions = ['']
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                samples = model.batchify_sample(
                    LQ_tensor, captions, num_steps=args.edm_steps, restoration_scale=args.s_stage1,
                    s_churn=args.s_churn, s_noise=args.s_noise, cfg_scale=args.s_cfg, control_scale=args.s_stage2,
                    seed=args.seed, num_samples=args.num_samples, p_p=args.a_prompt, n_p=args.n_prompt,
                    color_fix_type=args.color_fix_type, use_linear_CFG=args.linear_CFG,
                    use_linear_control_scale=args.linear_s_stage2, cfg_scale_start=args.spt_linear_CFG,
                    control_scale_start=args.spt_linear_s_stage2
                )
        output_img = Tensor2PIL(samples[0], h0, w0).resize(target_size)
        output_img.save(image_path)
        logger.info(f"Saved image at {image_path}")
    except Exception as e:
        logger.info(f"Error processing {image_path}: {e}")


def worker(gpu_id, bundle_list, target_sizes, args):
    """每个GPU独立处理部分数据"""
    logger = setup_logger(gpu_id,log_dir=args.log_dir)
    logger.info(f"Starting processing on GPU {gpu_id}")

    torch.cuda.set_device(gpu_id)  # 设置当前进程的GPU
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    try:
        model = create_SUPIR_model('options/SUPIR_v0.yaml', SUPIR_sign=args.SUPIR_sign)
        if args.loading_half_params:
            model = model.half()
        if args.use_tile_vae:
            model.init_tile_vae(encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size)
        model.ae_dtype = convert_dtype(args.ae_dtype)
        model.model.dtype = convert_dtype(args.diff_dtype)
        model = model.to(device)
        logger.info(f"Model initialized on GPU {gpu_id}")
    except Exception as e:
        logger.info(f"Error initializing model on GPU {gpu_id}: {e}")
        return
    tqdm_logger = TqdmToLogger(logger, logging.INFO)
    for trainlist_json in tqdm(bundle_list, desc=f"GPU {gpu_id}",file = tqdm_logger):
        try:
            #logger.info(f"Processing bundle: {trainlist_json}")
            if isinstance(trainlist_json, dict):
                data = trainlist_json
            else:
                with open(trainlist_json, 'r') as f:
                    data = json.load(f)

            for cam in CAM:
                file_info = data['camera_infos'][cam]
                target_size = target_sizes.get(cam, (1536, 864))
                filename = file_info['filename']
                if 'auto-oss:s3://sdc3-gt-qc-2/pap_finn/' in filename:
                    file_path = filename.replace('auto-oss:s3://sdc3-gt-qc-2/pap_finn/', './magicdrive-log/')
                elif "auto-oss:s3://sdc3-gt-qc-2/sdc3_finn/" in filename:
                    file_path = filename.replace('auto-oss:s3://sdc3-gt-qc-2/sdc3_finn/', './magicdrive-log/')
                else:
                    logger.info(f"Unknown path {filename}")
                    continue
                process_image(file_path, target_size, model, device,logger,args)
        except Exception as e:
            logger.info(f"Error processing bundle: {e}")


if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--process", type=int, default=-1)
    parser.add_argument("--num_split", type=int, default=2)
    parser.add_argument("--log_dir", type=str, default="./logs")
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


    # 数据准备
    input_folder = args.input
    if os.path.exists(os.path.join(input_folder, 'new_trainlist')):
        trainlist_path = os.path.join(input_folder, 'new_trainlist')
        bundle_list = os.listdir(trainlist_path)
        bundle_list = [os.path.join(trainlist_path, trainlist) for trainlist in bundle_list if
                       trainlist.endswith('.json')]
    else:
        with open(args.input, 'rb') as f:
            bundle_list = [json.loads(line) for line in f]

    subset = len(bundle_list) // int(args.num_split)
    if args.process==-1:
        bundle_list = bundle_list
    elif args.process == int(args.num_split) - 1:
        bundle_list = bundle_list[args.process * subset:]
    else:
        bundle_list = bundle_list[args.process * subset:(args.process + 1) * subset]
    # 划分数据
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # 动态分配chunk，确保最后一个chunk包含剩余的数据
    chunk_size = len(bundle_list) // num_gpus
    bundle_chunks = [bundle_list[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus - 1)]
    bundle_chunks.append(bundle_list[(num_gpus - 1) * chunk_size:])  # 最后一块包含剩余数据

    # 启动多进程
    mp.set_start_method("spawn")  # 必须设置多进程启动方式
    processes = []
    for gpu_id, chunk in enumerate(bundle_chunks):
        p = mp.Process(target=worker, args=(gpu_id, chunk, WH_TO_UPSCALE, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
