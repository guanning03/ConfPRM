import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 加载模型和tokenizer
print('正在加载模型...')
model_name = '/home/zgn/.cache/hf_models/Qwen/Qwen2.5-Math-7B'
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

# 读取数据
with open('/home/zgn/conf_prm/1226_conf_analysis_aime/selected_samples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def find_boxed_content(text):
    """找到最后一个 \\boxed{} 中的内容"""
    # 匹配 \boxed{...}，需要处理嵌套大括号
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = list(re.finditer(pattern, text))
    if matches:
        last_match = matches[-1]
        return last_match.group(1), last_match.start(), last_match.end()
    return None, None, None

def compute_boxed_token_prob(problem, response):
    """计算 \\boxed{} 中内容的 token 概率"""
    # 组装完整的 trajectory
    prompt = template.replace('[problem]', problem)
    full_text = prompt + response
    
    # Tokenize
    inputs = tokenizer(full_text, return_tensors='pt', return_offsets_mapping=True)
    input_ids = inputs['input_ids'].to(model.device)
    offset_mapping = inputs['offset_mapping'][0].tolist()  # [(start, end), ...]
    
    # 找到 boxed 内容在原文本中的位置
    boxed_content, boxed_start_char, boxed_end_char = find_boxed_content(full_text)
    if boxed_content is None:
        return None, None, "No \\boxed{} found"
    
    # 找到 boxed 内容开头对应的 token 位置（跳过 \boxed{ 本身）
    # 我们需要找到 \boxed{ 之后，内容开始的位置
    boxed_pattern = r'\\boxed\{'
    match = None
    for m in re.finditer(boxed_pattern, full_text):
        if m.end() <= boxed_end_char:
            match = m
    
    if match is None:
        return None, None, "Could not locate \\boxed{"
    
    # boxed 内容开始的字符位置（\boxed{ 之后）
    content_start_char = match.end()
    # boxed 内容结束的字符位置（} 之前）
    content_end_char = boxed_end_char - 1  # 减去最后的 }
    
    # 找到对应的 token 范围
    content_token_start = None
    content_token_end = None
    
    for idx, (start, end) in enumerate(offset_mapping):
        if start == end == 0:  # 特殊 token
            continue
        if content_token_start is None and end > content_start_char:
            content_token_start = idx
        if start < content_end_char:
            content_token_end = idx + 1
    
    if content_token_start is None or content_token_end is None:
        return None, None, "Could not map boxed content to tokens"
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]
    
    # 计算每个 token 的概率
    probs = torch.softmax(logits[0], dim=-1)  # [seq_len, vocab_size]
    
    # 获取 boxed 内容中每个 token 的概率
    # 注意：位置 i 的 logits 预测的是位置 i+1 的 token
    token_probs = []
    for i in range(content_token_start, content_token_end):
        # 位置 i-1 的 logits 预测位置 i 的 token
        token_id = input_ids[0, i].item()
        prob = probs[i - 1, token_id].item()
        token_text = tokenizer.decode([token_id])
        token_probs.append({
            'token_id': token_id,
            'token_text': token_text,
            'prob': prob
        })
    
    # 计算整体概率（所有 token 概率的乘积）
    if token_probs:
        overall_prob = 1.0
        for tp in token_probs:
            overall_prob *= tp['prob']
    else:
        overall_prob = None
    
    return boxed_content, token_probs, overall_prob

# 处理所有样本
results = []

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
    for resp in problem_data['correct_responses']:
        response = resp['response']
        boxed_content, token_probs, overall_prob = compute_boxed_token_prob(problem, response)
        
        problem_result['correct_responses'].append({
            'response_id': resp['response_id'],
            'prediction': resp['prediction'],
            'boxed_content': boxed_content,
            'token_probs': token_probs,
            'overall_prob': overall_prob
        })
    
    # 处理错误的 responses
    for resp in problem_data['wrong_responses']:
        response = resp['response']
        boxed_content, token_probs, overall_prob = compute_boxed_token_prob(problem, response)
        
        problem_result['wrong_responses'].append({
            'response_id': resp['response_id'],
            'prediction': resp['prediction'],
            'boxed_content': boxed_content,
            'token_probs': token_probs,
            'overall_prob': overall_prob
        })
    
    results.append(problem_result)

# 保存结果
output_path = '/home/zgn/conf_prm/1226_conf_analysis_aime/boxed_probs.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f'\n结果已保存到: {output_path}')

# 打印一些统计信息
print('\n=== 统计信息 ===')
for problem_result in results[:3]:  # 只打印前3道题
    print(f"\nProblem {problem_result['problem_id']} (Gold: {problem_result['gold']}):")
    
    print("  正确响应:")
    for resp in problem_result['correct_responses'][:2]:
        print(f"    Response {resp['response_id']}: boxed={resp['boxed_content']}, overall_prob={resp['overall_prob']:.6f}" if isinstance(resp['overall_prob'], float) else f"    Response {resp['response_id']}: {resp['overall_prob']}")
    
    print("  错误响应:")
    for resp in problem_result['wrong_responses'][:2]:
        print(f"    Response {resp['response_id']}: boxed={resp['boxed_content']}, overall_prob={resp['overall_prob']:.6f}" if isinstance(resp['overall_prob'], float) else f"    Response {resp['response_id']}: {resp['overall_prob']}")

