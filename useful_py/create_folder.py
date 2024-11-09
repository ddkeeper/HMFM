import os

def create_folder_structure(parent_folders, subfolder_file):
    # 读取子文件夹名称
    with open(subfolder_file, 'r') as f:
        subfolders = [line.strip() for line in f if line.strip()]
    
    # 为每个父文件夹创建结构
    for parent in parent_folders:
        if not os.path.exists(parent):
            os.makedirs(parent)
        
        # 在父文件夹中创建子文件夹
        for subfolder in subfolders:
            subfolder_path = os.path.join(parent, subfolder)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
                print(f"创建文件夹: {subfolder_path}")
            else:
                print(f"文件夹已存在: {subfolder_path}")

# 使用示例
#parent_folders = ["results_blca", "results_luad", "results_ucec"]  # 可以根据需要修改父文件夹列表
parent_folders = ["results_blca"]
subfolder_file = "baseline_name.txt"  # 包含子文件夹名称的txt文件

create_folder_structure(parent_folders, subfolder_file)
