import pandas as pd

# 步骤0: 读取原始CSV文件
tot_info_csv_path = "/home/yinwendong/MCAT-master/splits/1foldcv/tcga_kirc/tot_info_new_filled.csv"
#tot_info_csv_path = 'tot_info_try.csv'
kirc_clinical_csv_path = 'kirc_tcga_pan_can_atlas_2018_clinical_data.csv'

tot_info_df = pd.read_csv(tot_info_csv_path)
kirc_clinical_df = pd.read_csv(kirc_clinical_csv_path)

# 步骤1: 值映射修改grade列
grade_mapping = {'G4': 4, 'G3': 3, 'G2': 2, 'G1': 1}
tot_info_df['grade'] = tot_info_df['grade'].map(grade_mapping)

# 步骤2: 获取10q Status到9p Status的索引范围，并对范围内的Status列进行值映射
#status_columns_indices = []
columns = tot_info_df.columns
begin = columns.get_loc('10q Status')
end = columns.get_loc('9p Status')
for index in range(begin, end + 1):
    #status_columns_indices.append(index)
    tot_info_df.iloc[:, index] = tot_info_df.iloc[:, index].map({'Gained': 1, 'Not Called': -1, 'Lost': 0, 'NA': 0})

# 步骤3: 在case_id列之后添加dis_label列并初始化为0
case_id_index = tot_info_df.columns.get_loc('case_id')
tot_info_df.insert(case_id_index + 1, 'dis_label', 0)

# 步骤4: 在grade列之后添加censorship列并根据条件赋值
grade_index = tot_info_df.columns.get_loc('grade')
censorship_mapping = kirc_clinical_df.set_index('Patient ID')['Progression Free Status'].map(lambda x: 1 if x == 'CENSORED' else 0)
tot_info_df.insert(grade_index + 1, 'censorship', tot_info_df['case_id'].map(censorship_mapping))

# 步骤5: 在survival_months列之后添加10p Status列并根据条件赋值
dic = {'Gained': 1, 'Not Called': -1, 'Lost': 0, 'NA': 0}
survival_months_index = tot_info_df.columns.get_loc('survival_months')
Status_10p_mapping = kirc_clinical_df.set_index('Patient ID')['10p Status'].map(dic)
tot_info_df.insert(survival_months_index + 1, '10p Status', tot_info_df['case_id'].map(Status_10p_mapping))

# 步骤6: 在9p Status列之后添加9q Status列并根据条件赋值
dic = {'Gained': 1, 'Not Called': -1, 'Lost': 0, 'NA': 0}
Status_9p_index = tot_info_df.columns.get_loc('9p Status')
Status_9q_mapping = kirc_clinical_df.set_index('Patient ID')['9q Status'].map(dic)
tot_info_df.insert(Status_9p_index + 1, '9q Status', tot_info_df['case_id'].map(Status_9q_mapping))

# 步骤7: 保存到新的CSV文件
tot_info_new_csv_path = 'tot_info_new.csv'
tot_info_df.to_csv(tot_info_new_csv_path, index=False)

print('数据处理完成，并已保存至:', tot_info_new_csv_path)
