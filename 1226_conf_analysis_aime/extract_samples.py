from datasets import load_dataset
from collections import defaultdict
import json
import random

# 设置随机种子以确保可重复性
random.seed(42)

# 加载数据集
print('正在加载数据集...')
dataset = load_dataset('guanning-ai/Qwen2.5-7B_aime2425_4096_train0test64')
print('加载完成！')

test_data = dataset['test']

# 按 problem_id 分组收集正确和错误的 response
correct_responses = defaultdict(list)
wrong_responses = defaultdict(list)
problem_info = {}  # 存储题目信息

for sample in test_data:
    pid = sample['problem_id']
    
    # 保存题目信息
    if pid not in problem_info:
        problem_info[pid] = {
            'problem': sample['problem'],
            'gold': sample['gold']
        }
    
    # 必须包含 \boxed{}
    if '\\boxed{' not in sample['response']:
        continue
    
    response_data = {
        'response_id': sample['response_id'],
        'response': sample['response'],
        'prediction': sample['prediction'],
        'num_tokens': sample['num_tokens']
    }
    
    if sample['correctness']:
        correct_responses[pid].append(response_data)
    else:
        # 错误的response需要token数量不超过3000
        if sample['num_tokens'] <= 3000:
            wrong_responses[pid].append(response_data)

# 筛选满足条件的题目并生成结果
result = []

for pid in range(64):
    correct_list = correct_responses[pid]
    wrong_list = wrong_responses[pid]
    
    # 检查是否至少有5个正确和5个错误的response
    if len(correct_list) >= 5 and len(wrong_list) >= 5:
        # 随机选择5个正确和5个错误的response
        selected_correct = random.sample(correct_list, 5)
        selected_wrong = random.sample(wrong_list, 5)
        
        result.append({
            'problem_id': pid,
            'problem': problem_info[pid]['problem'],
            'gold': problem_info[pid]['gold'],
            'correct_responses': selected_correct,
            'wrong_responses': selected_wrong
        })
        print(f'Problem {pid}: 满足条件 (正确: {len(correct_list)}, 错误(token<=3000): {len(wrong_list)})')
    else:
        print(f'Problem {pid}: 跳过 (正确: {len(correct_list)}, 错误(token<=3000): {len(wrong_list)})')

# 保存为JSON文件
output_path = '/home/zgn/conf_prm/1226_conf_analysis_aime/selected_samples.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f'\n共筛选出 {len(result)} 道题目')
print(f'结果已保存到: {output_path}')

