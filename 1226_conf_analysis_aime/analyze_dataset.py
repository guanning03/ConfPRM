from datasets import load_dataset
from collections import defaultdict

# 加载数据集
print('正在加载数据集...')
dataset = load_dataset('guanning-ai/Qwen2.5-7B_aime2425_4096_train0test64')
print('加载完成！')

# 统计每个 problem_id 的正确率
test_data = dataset['test']

# 用字典统计每个 problem_id 的正确数和总数
correct_count = defaultdict(int)
total_count = defaultdict(int)

for sample in test_data:
    pid = sample['problem_id']
    if 0 <= pid <= 63:
        total_count[pid] += 1
        if sample['correctness']:
            correct_count[pid] += 1

# 打印每个 problem_id 的正确率
print('\n=== Problem ID 0-63 的正确率 ===')
print(f'{"Problem ID":<12} {"正确数":<10} {"总数":<10} {"正确率":<10}')
print('-' * 45)

total_correct = 0
total_samples = 0

for pid in range(64):
    if total_count[pid] > 0:
        rate = correct_count[pid] / total_count[pid] * 100
        print(f'{pid:<12} {correct_count[pid]:<10} {total_count[pid]:<10} {rate:.2f}%')
        total_correct += correct_count[pid]
        total_samples += total_count[pid]

print('-' * 45)
if total_samples > 0:
    overall_rate = total_correct / total_samples * 100
    print(f'{"总计":<12} {total_correct:<10} {total_samples:<10} {overall_rate:.2f}%')

