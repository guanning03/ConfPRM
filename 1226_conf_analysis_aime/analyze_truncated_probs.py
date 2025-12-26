import json
import torch
import re
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# 加载模型和tokenizer
print('正在加载模型...')
model_name = '/data/user_data/jgai/cache/verl-checkpoints/dynamic-coeff-math-7b-1225/grpo_ap0.01_an0.01_seed0/global_step_500/actor/huggingface'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True
)
model.eval()
print('模型加载完成！')

# 读取模板
template = """<|im_start|>system
Please reason step by step and put the final answer in \\boxed{}. <|im_end|>
<|im_start|>user
[problem] Let's think step by step and output the final answer within \\boxed{}. <|im_end|>
<|im_start|>assistant
"""

# 直接给出答案的后缀
answer_suffix = """Thus, the final answer is:
\\[
\\boxed{[gold]}
\\]"""

# 读取数据
with open('./1226_conf_analysis_aime/selected_samples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def get_truncated_response(response, percentage):
    """按 \\n\\n 分段，取前 percentage% 的段落"""
    if percentage == 0:
        return ""
    
    # 按 \n\n 分段
    paragraphs = response.split('\n\n')
    
    # 计算要取的段落数
    total_paragraphs = len(paragraphs)
    num_to_take = max(1, int(total_paragraphs * percentage / 100))
    
    if percentage == 100:
        num_to_take = total_paragraphs
    
    # 取前 num_to_take 个段落
    truncated = '\n\n'.join(paragraphs[:num_to_take])
    return truncated

def compute_gold_prob_for_truncated(problem, truncated_response, gold):
    """计算截断后 response + gold 答案的概率"""
    # 组装完整的 trajectory
    prompt = template.replace('[problem]', problem)
    suffix = answer_suffix.replace('[gold]', gold)
    
    if truncated_response:
        full_text = prompt + truncated_response + "\n\n" + suffix
    else:
        full_text = prompt + suffix
    
    # Tokenize
    inputs = tokenizer(full_text, return_tensors='pt', return_offsets_mapping=True)
    input_ids = inputs['input_ids'].to(model.device)
    offset_mapping = inputs['offset_mapping'][0].tolist()
    
    # 找到 \boxed{ 之后 gold 内容的位置
    boxed_pattern = r'\\boxed\{'
    match = None
    for m in re.finditer(boxed_pattern, full_text):
        match = m  # 取最后一个匹配
    
    if match is None:
        return None
    
    # gold 内容开始的字符位置
    content_start_char = match.end()
    content_end_char = full_text.find('}', content_start_char)
    
    # 找到对应的 token 范围
    content_token_start = None
    content_token_end = None
    
    for idx, (start, end) in enumerate(offset_mapping):
        if start == end == 0:
            continue
        if content_token_start is None and end > content_start_char:
            content_token_start = idx
        if start < content_end_char:
            content_token_end = idx + 1
    
    if content_token_start is None or content_token_end is None:
        return None
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    # 计算每个 token 的概率
    probs = torch.softmax(logits[0], dim=-1)
    
    # 获取 gold 内容中每个 token 的概率
    overall_prob = 1.0
    for i in range(content_token_start, content_token_end):
        token_id = input_ids[0, i].item()
        prob = probs[i - 1, token_id].item()
        overall_prob *= prob
    
    return overall_prob

# 百分比列表
percentages = [0, 25, 50, 75, 100]

# 处理所有样本并收集数据
all_results = []

for problem_data in tqdm(data, desc="Processing problems"):
    problem_id = problem_data['problem_id']
    problem = problem_data['problem']
    gold = problem_data['gold']
    
    problem_result = {
        'problem_id': problem_id,
        'gold': gold,
        'correct_responses': [],
        'wrong_responses': []
    }
    
    # 处理正确的 responses
    for resp in tqdm(problem_data['correct_responses'], desc=f"Problem {problem_id} correct", leave=False):
        response = resp['response']
        response_id = resp['response_id']
        
        probs_at_percentages = []
        for pct in percentages:
            truncated = get_truncated_response(response, pct)
            prob = compute_gold_prob_for_truncated(problem, truncated, gold)
            probs_at_percentages.append(prob)
        
        problem_result['correct_responses'].append({
            'response_id': response_id,
            'probs': probs_at_percentages
        })
    
    # 处理错误的 responses
    for resp in tqdm(problem_data['wrong_responses'], desc=f"Problem {problem_id} wrong", leave=False):
        response = resp['response']
        response_id = resp['response_id']
        
        probs_at_percentages = []
        for pct in percentages:
            truncated = get_truncated_response(response, pct)
            prob = compute_gold_prob_for_truncated(problem, truncated, gold)
            probs_at_percentages.append(prob)
        
        problem_result['wrong_responses'].append({
            'response_id': response_id,
            'probs': probs_at_percentages
        })
    
    all_results.append(problem_result)

# 保存结果
output_json_path = './1226_conf_analysis_aime/truncated_probs.json'
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f'结果已保存到: {output_json_path}')

# 创建输出目录
output_dir = './1226_conf_analysis_aime/plots'
os.makedirs(output_dir, exist_ok=True)

# 画图
print('\n正在生成图表...')
for result in all_results:
    problem_id = result['problem_id']
    gold = result['gold']
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'Problem {problem_id} (Gold: {gold})', fontsize=16)
    
    # 上面一行：正确的 responses
    for i, resp_data in enumerate(result['correct_responses']):
        ax = axes[0, i]
        response_id = resp_data['response_id']
        probs = resp_data['probs']
        
        ax.plot(percentages, probs, 'go-', linewidth=2, markersize=8)
        ax.set_xlabel('Truncation %')
        ax.set_ylabel('Probability')
        ax.set_title(f'Correct - Response {response_id}')
        ax.set_xticks(percentages)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    # 下面一行：错误的 responses
    for i, resp_data in enumerate(result['wrong_responses']):
        ax = axes[1, i]
        response_id = resp_data['response_id']
        probs = resp_data['probs']
        
        ax.plot(percentages, probs, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('Truncation %')
        ax.set_ylabel('Probability')
        ax.set_title(f'Wrong - Response {response_id}')
        ax.set_xticks(percentages)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'problem_{problem_id}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'已保存: {output_path}')

print(f'\n所有图表已保存到: {output_dir}')

