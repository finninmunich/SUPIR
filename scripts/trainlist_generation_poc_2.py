import argparse
import os
import pickle
import json
import sys
sys.path.append('./')
import random
import copy
import shutil
from tqdm import tqdm
class ImageProcessor:
    def __init__(self, args):
        self.auto_collect_folder = args.auto_collect_folder
        self.trainlist_file = args.trainlist_file
        self.image_dict_list = []
        self.mapping = ['center_camera_fov120', 'center_camera_fov30', 'left_front_camera', 'right_front_camera',
                        'left_rear_camera', 'right_rear_camera', 'rear_camera', 'front_camera_fov195',
                        'left_camera_fov195',
                        'right_camera_fov195', 'rear_camera_fov195']
        self.index_list = []
        self.meta_dict = {}
        self.s3_prefix = 'auto-oss:s3://sdc3-gt-qc-2/aigc/N01-002/delivery/'
        self.local_prefix = '/finn/finn/DATA/sensetime-data/delivery/20240904/part1/'
    def load_trainlist(self,trainlist):
        data=[]
        with open(trainlist, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    def run(self):
        data = self.load_trainlist(self.trainlist_file)
        deliver_data=[]
        for _data in tqdm(data):
            parse = _data['camera_infos']['center_camera_fov30']['filename'].split('/')[-4]
            new_data = copy.deepcopy(_data)
            ori_data = copy.deepcopy(_data)
            new_data['aigc'] = True
            HR_delivery_dir = os.path.join(self.auto_collect_folder, parse, 'camera_HR_delivery')
            for cam in self.mapping:
                #_parser_date = _data['camera_infos'][cam]['filename'].split('/')[-4].split('_pilotGtParser')[0]
                img_name = _data['camera_infos'][cam]['filename'].split('/')[-1]
                img_dir = os.path.join(HR_delivery_dir, cam,img_name.replace('.jpg','.png'))
                assert os.path.exists(img_dir), f"Path {img_dir} does not exist"
                delivery_img_dir = os.path.join(self.s3_prefix,img_dir.split('/delivery/')[-1])
                new_data['camera_infos'][cam]['filename'] = delivery_img_dir
            deliver_data.append(new_data)
            # add rainy day
            raw_data = ori_data['raw_data']
            for cam in self.mapping:
                new_filename = ori_data['camera_infos'][cam]['filename']
                ori_data['camera_infos'][cam]['filename'] = raw_data + '/' + '/'.join(new_filename.split('/')[-3:])
            ori_data['aigc'] = False
            deliver_data.append(ori_data)
        print(f"total {len(deliver_data)} bundle processed.")
        with open(args.output, 'w') as f:
            for i, bundle in enumerate(deliver_data):
                # 判断是否为最后一个元素
                if i < len(deliver_data) - 1:
                    f.write(json.dumps(bundle) + '\n')
                else:
                    f.write(json.dumps(bundle))  # 最后一行不加换行符


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process generated images and save to trainlist file.")
    parser.add_argument('--auto_collect_folder', '-at', type=str, help='Folder containing images')
    parser.add_argument('--trainlist_file', '-t_list', default='data/sensetime-data/2024-07-21/sensetime-train-v2.pkl', type=str,
                        help='train list for selected folder')
    parser.add_argument('--output', '-o', default='trainlist_rainy.txt', type=str, help='Output file name')
    args = parser.parse_args()
    processor = ImageProcessor(args)
    processor.run()
