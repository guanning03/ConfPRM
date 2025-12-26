from datasets import load_dataset
import json

# 下载数据集
print('正在下载数据集...')
dataset = load_dataset('guanning-ai/Qwen2.5-7B_aime2425_4096_train0test64')
print('下载完成！')

# 查看数据集信息
print('\n=== 数据集整体结构 ===')
print(dataset)

# 查看各个split
for split_name in dataset:
    print(f'\n=== {split_name} split ===')
    print(f'样本数量: {len(dataset[split_name])}')
    print(f'列名: {dataset[split_name].column_names}')
    print(f'特征: {dataset[split_name].features}')
    
    # 查看第一个样本
    print(f'\n--- {split_name} 第一个样本示例 ---')
    sample = dataset[split_name][0]
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 500:
            print(f'{key}: {value[:500]}...(截断)')
        elif isinstance(value, list) and len(value) > 3:
            print(f'{key}: {value[:3]}...(共{len(value)}项)')
        else:
            print(f'{key}: {value}')

