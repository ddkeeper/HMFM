import csv
import os

def check_files_exist(csv_file, folder):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        column_counts = {'train': 0, 'val': 0, 'test': 0}
        missing_files = {'train': 0, 'val': 0, 'test': 0}
        for row in reader:
            for column in ['train', 'val', 'test']:
                case = row[column]
                if not case:
                    continue
                column_counts[column] += 1
                file_path = os.path.join(folder, case + '.pt')
                if not os.path.exists(file_path):
                    print(case)
                    print(f"文件 {case} 在 {column} 列中不存在")
                    missing_files[column] += 1
        
        for column in ['train', 'val', 'test']:
            print(f"{column} 列中存在 {column_counts[column]} 个元素")
            print(f"{column} 列中缺失 {missing_files[column]} 个文件")

# 使用示例
csv_file = "/home/wangshijin/projects/MoME/splits/1foldcv/tcga_brca/splits_0.csv"
folder = "/home/wangshijin/projects/DEQFusion/experiments/BRCA/features/tcga-brca/mcat_survival_path_features/"
check_files_exist(csv_file, folder)