import os
import shutil
import argparse

def main(source_root_dir, target_dir):
    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)

    # 获取所有相机文件夹的名称
    camera_folders = [
        'center_camera_fov30', 'center_camera_fov120', 'left_front_camera',
        'left_rear_camera', 'rear_camera', 'right_front_camera',
        'right_rear_camera', 'front_camera_fov195', 'left_camera_fov195',
        'rear_camera_fov195', 'right_camera_fov195'
    ]

    # 遍历每个相机文件夹
    for camera in camera_folders:
        camera_folder_path = os.path.join(source_root_dir, camera)

        # 检查相机文件夹是否存在
        if not os.path.exists(camera_folder_path):
            print(f"warning: folder {camera_folder_path} doesn't exist，skip。")
            continue

        # 遍历相机文件夹中的每个图像文件
        for image_file in os.listdir(camera_folder_path):
            if image_file.endswith('.jpg'):
                # 构造源文件路径和目标文件路径
                source_file_path = os.path.join(camera_folder_path, image_file)
                target_file_name = f"{camera}_{image_file}"
                target_file_path = os.path.join(target_dir, target_file_name)

                # 将图像复制到目标文件夹并重命名
                shutil.copy(source_file_path, target_file_path)
                print(f"copy && rename {source_file_path} to {target_file_path}")

    print("Process Done")

if __name__ == "__main__":
    # 设置参数解析
    parser = argparse.ArgumentParser(description="rename images")
    parser.add_argument("--source", type=str, required=True, help="source root")
    parser.add_argument("--target", type=str, required=True, help="target root")

    # 解析参数
    args = parser.parse_args()

    # 调用主函数
    main(args.source, args.target)
