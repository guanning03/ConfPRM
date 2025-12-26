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

# 直接给出答案的 response 模板
direct_answer_template = """Thus, the final answer is:
\\[
\\boxed{[gold]}
\\]"""

# 读取数据
with open('/home/zgn/conf_prm/1226_conf_analysis_aime/selected_samples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def compute_gold_token_prob(problem, gold):
    """计算直接给出 gold 答案时，gold token 的概率"""
    # 组装完整的 trajectory
    prompt = template.replace('[problem]', problem)
    response = direct_answer_template.replace('[gold]', gold)
    full_text = prompt + response
    
    # Tokenize
    inputs = tokenizer(full_text, return_tensors='pt', return_offsets_mapping=True)
    input_ids = inputs['input_ids'].to(model.device)
    offset_mapping = inputs['offset_mapping'][0].tolist()  # [(start, end), ...]
    
    # 找到 \boxed{ 之后 gold 内容的位置
    # 在 full_text 中找到 \boxed{gold} 的位置
    boxed_pattern = r'\\boxed\{'
    match = None
    for m in re.finditer(boxed_pattern, full_text):
        match = m  # 取最后一个匹配
    
    if match is None:
        return None, None, "Could not locate \\boxed{"
    
    # gold 内容开始的字符位置（\boxed{ 之后）
    content_start_char = match.end()
    # gold 内容结束的字符位置（} 之前）
    # 找到对应的 }
    content_end_char = full_text.find('}', content_start_char)
    
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
        return None, None, "Could not map gold content to tokens"
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]
    
    # 计算每个 token 的概率
    probs = torch.softmax(logits[0], dim=-1)  # [seq_len, vocab_size]
    
    # 获取 gold 内容中每个 token 的概率
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
    
    return gold, token_probs, overall_prob

# 处理所有样本
results = []

for problem_data in tqdm(data, desc="Processing problems"):
    problem_id = problem_data['problem_id']
    problem = problem_data['problem']
    gold = problem_data['gold']
    
    # 计算 gold 的概率
    boxed_content, token_probs, overall_prob = compute_gold_token_prob(problem, gold)
    
    results.append({
        'problem_id': problem_id,
        'gold': gold,
        'boxed_content': boxed_content,
        'token_probs': token_probs,
        'overall_prob': overall_prob
    })

# 保存结果
output_path = '/home/zgn/conf_prm/1226_conf_analysis_aime/gold_probs.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f'\n结果已保存到: {output_path}')

# 打印结果
print('\n=== Gold 概率统计 ===')
for result in results:
    prob_str = f"{result['overall_prob']:.6f}" if isinstance(result['overall_prob'], float) else str(result['overall_prob'])
    token_info = ""
    if isinstance(result['token_probs'], list):
        tokens = [f"{tp['token_text']}({tp['prob']:.4f})" for tp in result['token_probs']]
        token_info = " | ".join(tokens)
    print(f"Problem {result['problem_id']}: gold={result['gold']}, overall_prob={prob_str}")
    if token_info:
        print(f"  Tokens: {token_info}")

