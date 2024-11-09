import csv
from collections import defaultdict

def analyze_expert_choices(samples_expert_choices, output_file='amfm_expert_choices.csv'):
    # 准备 CSV 文件
    fieldnames = ['case_id'] + [f'patho_{i}' for i in range(4)] + [f'genom_{i}' for i in range(4)] + [f'fuse_{i}' for i in range(4)]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 用于汇总所有样本的计数器
        total_counts = {
            'patho': defaultdict(int),
            'genom': defaultdict(int),
            'fuse': defaultdict(int)
        }

        # 用于计算最后一行汇总数据的字典
        summary_row = {field: 0 for field in fieldnames if field != 'case_id'}

        for case_id, choices in samples_expert_choices.items():
            row = {'case_id': case_id}
            counts = {
                'patho': defaultdict(int),
                'genom': defaultdict(int),
                'fuse': defaultdict(int)
            }

            for submicrobatch in choices:
                counts['patho'][submicrobatch["corresponding_net_id_patho1"]] += 1
                counts['genom'][submicrobatch["corresponding_net_id_genom1"]] += 1
                counts['fuse'][submicrobatch["corresponding_net_id_fuse"]] += 1

            # 填充行数据
            for expert_type in ['patho', 'genom', 'fuse']:
                for i in range(4):  # 假设每种类型最多有4个专家
                    count = counts[expert_type][i]
                    field = f'{expert_type}_{i}'
                    row[field] = count
                    total_counts[expert_type][i] += count
                    summary_row[field] += count  # 累加到汇总行

            writer.writerow(row)

        # 写入汇总行
        summary_row['case_id'] = 'Total'
        writer.writerow(summary_row)

    print(f"Individual sample data and summary row have been written to {output_file}")

    # 打印汇总结果
    print("\nOverall expert selection summary:")
    for expert_type in ['patho', 'genom', 'fuse']:
        print(f"\n{expert_type.capitalize()} experts:")
        total = sum(total_counts[expert_type].values())
        for expert_id, count in total_counts[expert_type].items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  Expert {expert_id}: {count} times ({percentage:.2f}%)")

# 使用方法
#analyze_expert_choices(samples_expert_choices)
