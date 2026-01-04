import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import re

MODEL_PATH = "model_path"
INPUT_FILES = [
    "question.txt",
]
OUTPUT_PREFIX = "answers_"
BATCH_SIZE = 1
GEN_LENGTH = 40  
STEPS = 40       
BLOCK_LENGTH = 8 
CFG_SCALE = 1.0
def add_gumbel_noise(logits, temperature):
    """为 logits 添加 Gumbel 噪声以进行采样。"""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    """计算每一步需要揭示（unmask）的 token 数量。"""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    if steps == 0: return torch.zeros(mask_num.size(0), 0, device=mask_index.device, dtype=torch.int64)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def generate_answer(model, prompts, tokenizer, steps, gen_length, block_length, temperature=0.,
                    cfg_scale=0., remasking='low_confidence', mask_id=126336):
    """为一批 prompts 生成答案的核心函数。"""
    batch_size, prompt_length = prompts.shape
    x = torch.full((batch_size, prompt_length + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt_length] = prompts.clone()
    
    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    prompt_index[:, :prompt_length] = True

    assert gen_length % block_length == 0, "gen_length 必须是 block_length 的整数倍"
    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        steps = (steps // num_blocks) * num_blocks if (steps // num_blocks) > 0 else num_blocks
        print(f"提示：steps已自动调整为{steps}以匹配分块设置。")

    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start_index = prompt_length + num_block * block_length
        block_end_index = prompt_length + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start_index:block_end_index] == mask_id)
        if not block_mask_index.any():
            continue

        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            if not mask_index.any():
                break
            
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            scores = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(scores, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand_like(x0, dtype=torch.float32)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end_index:] = -float('inf')
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float('inf'))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                block_confidence = confidence[j, block_start_index:block_end_index]
                k = min(num_transfer_tokens[j, i].item(), (block_confidence > -float('inf')).sum().item())
                if k == 0: continue

                _, select_indices_in_block = torch.topk(block_confidence, k=k)
                select_indices = select_indices_in_block + block_start_index
                transfer_index[j, select_indices] = True
            
            x = torch.where(transfer_index, x0, x)
    return x

def run_inference_on_file(model, tokenizer, device, input_file, output_file):
    """读取文件，批量生成答案并保存。"""
    print(f"\n--- 正在处理文件: {input_file} ---")
    if not os.path.exists(input_file):
        print(f"警告: 未找到输入文件 '{input_file}', 已跳过。")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"找到 {len(questions)} 个问题。开始使用批量大小 {BATCH_SIZE} 生成答案...")
    
    remasking_strategy = 'low_confidence'
    print(f"为Base SFT模型，已指定使用 '{remasking_strategy}' remasking策略。")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(questions), BATCH_SIZE), desc=f"处理 {os.path.basename(input_file)}"):
            batch_questions = questions[i:i + BATCH_SIZE]
            
            inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            output_ids = generate_answer(
                model, inputs['input_ids'], tokenizer, 
                steps=STEPS, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH, 
                cfg_scale=CFG_SCALE, remasking=remasking_strategy
            )

            answers = tokenizer.batch_decode(output_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            for idx, answer in enumerate(answers):
                original_question_index = i + idx
                clean_answer_for_print = answer.strip().replace('\n', ' ')
                print(f"  样本 {original_question_index+1}: 问题='{batch_questions[idx][:30]}...', 答案='{clean_answer_for_print[:50]}...'")
                f_out.write(answer.strip().replace('\n', ' ') + '\n')
                f_out.flush() 
    print(f"答案已成功保存到: {output_file}")

def main():
    """主函数"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        device = 'npu'
    else:
        device = 'cpu'
    print(f"使用的设备: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型路径 '{MODEL_PATH}' 不存在。")
        return

    print("正在加载模型和分词器...")
    try:
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    for input_file in INPUT_FILES:
        base_name = os.path.basename(input_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = f"{OUTPUT_PREFIX}{name_without_ext}(token).txt"
        run_inference_on_file(model, tokenizer, device, input_file, output_file)

if __name__ == "__main__":
    main()
