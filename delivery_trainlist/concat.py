
# files = [
#     './deliver_20240818_HR/2024_06_20_11_02_05_AutoCollect_trainlist.txt',
#     './deliver_20240818_HR/2024_06_20_11_02_05_AutoCollect_trainlist.txt',
#     './deliver_20240818_HR/2024_06_20_18_16_06_AutoCollect_trainlist.txt'
# ]
files = [
    './deliver_20240818_HR/deliver_20240818_20240624.txt',
    './deliver_20240818_HR/deliver_20240818_20240613.txt',
]
# 定义合并后的输出文件
# output_file = 'trainlist_evening.txt'
output_file = './deliver_20240818_HR/deliver_20240818.txt'

if __name__ == '__main__':
    # 打开输出文件，以写入模式
    with open(output_file, 'w') as outfile:
        # 逐个文件读取内容并写入输出文件
        for i, file in enumerate(files):
            with open(file, 'r') as infile:
                # 读取当前文件的内容并写入输出文件
                outfile.write(infile.read())
                if i < len(files) - 1:
                    # 在每个文件之间添加一个换行符（可选）
                    outfile.write('\n')
