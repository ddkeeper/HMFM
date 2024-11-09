import pandas as pd

# 读取splits文件和result文件
splits_df = pd.read_csv("/home/yinwendong/MCAT-master/splits/1foldcv/tcga_brca/s.csv")
result_df = pd.read_csv("/home/yinwendong/MCAT-master/dataset_csv/TCGA-BRCA/result.csv")


# 创建一个空的DataFrame来存储符合条件的数据
new_df = pd.DataFrame(columns=['train', 'val', 'test'])

# 遍历splits文件的train，val和test列
for column in ['test']:
    ids = splits_df[column].tolist()
    new_column = []
    cnt = 0
    print(len(ids))
    # 遍历每个ID
    for id in ids:
        # 在result文件中找到对应的行
        matching_row = result_df[result_df['case_id'] == id]
        
        # 检查label列的值是否为指定的三种类型
        if not matching_row.empty:
            label_value = matching_row['label'].values[0]
            if label_value in ['Infiltrating Ductal Carcinoma', 'Infiltrating Lobular Carcinoma', 'Medullary Carcinoma']:
                cnt += 1
                new_column.append(id)
        else:
            pass
            #new_column.append(id)
    print(cnt)
    new_df[column] = new_column

# 将结果写入到new文件
new_df.to_csv("/home/yinwendong/MCAT-master/splits/1foldcv/tcga_brca/splits_for_clam.csv", index=False)

