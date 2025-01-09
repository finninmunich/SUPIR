import os
import argparse
import json
from tqdm import tqdm
import copy

CAM = ['center_camera_fov120', 'center_camera_fov30', 'left_front_camera', 'right_front_camera',
       'left_rear_camera', 'right_rear_camera', 'rear_camera', 'front_camera_fov195',
       'left_camera_fov195',
       'right_camera_fov195', 'rear_camera_fov195']


def get_earliest_files(folder_path, num_files):
    # 获取文件夹中所有文件的完整路径
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                 os.path.isfile(os.path.join(folder_path, f))]

    # 获取文件的创建时间，并将文件路径和创建时间组合在一起
    files_with_ctime = [(file, os.path.getctime(file)) for file in all_files]

    # 按照创建时间进行排序
    files_with_ctime.sort(key=lambda x: x[1])

    # 选取最早的 num_files 个文件
    earliest_files = [file for file, _ in files_with_ctime[:num_files]]
    return earliest_files


def trainlist_generation(trainlist_path, args):
    delivery_trainlist = []
    for trainlist in tqdm(trainlist_path):
        with open( trainlist, 'r') as f:
            data_infos = json.load(f)
            original_data_infos = copy.deepcopy(data_infos)
            data_infos['gt_ann_info_path'] = data_infos['gt_ann_info_path'].replace(
                    '/pap_finn/', '/pap_finn/production-0918/pap_finn/')
            original_data_infos['gt_ann_info_path'] = original_data_infos['gt_ann_info_path'].replace('/pap_finn/', '/pap/')
            for cam in CAM:
                data_infos['camera_infos'][cam]['filename'] = data_infos['camera_infos'][cam]['filename'].replace(
                    '/pap_finn/', '/pap_finn/production-0918/pap_finn/')
                original_data_infos['camera_infos'][cam]['filename'] = original_data_infos['camera_infos'][cam][
                    'filename'].replace('/pap_finn/', '/pap/')
            delivery_trainlist.append(data_infos)
            delivery_trainlist.append(original_data_infos)
    print(f"total {len(delivery_trainlist)} bundle processed.")
    with open(args.output, 'w') as f:
        for i, bundle in enumerate(delivery_trainlist):
            # 判断是否为最后一个元素
            if i < len(delivery_trainlist) - 1:
                f.write(json.dumps(bundle) + '\n')
            else:
                f.write(json.dumps(bundle))  # 最后一行不加换行符



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the earliest files in a folder based on creation time.")
    parser.add_argument("--folder", '-i', type=str, help="Path to the folder.")
    parser.add_argument("--num_files", '-n', type=int, default=500,
                        help="Number of earliest files to select. Default is 500.")
    parser.add_argument("--output", '-o', type=str, help="output file path")

    args = parser.parse_args()
    folder_path = args.folder
    num_files = args.num_files

    # 获取最早的文件
    earliest_files = get_earliest_files(folder_path, num_files)
    trainlist_generation(earliest_files, args)
    # # 打印或处理最早的文件
    # for file in earliest_files:
    #     print(file)
