import json
import os

def evaluate_completion(
    completion: str,
    target: str,
    case_sensitive: bool = False,
) -> bool:

    target_list = [t.strip() for t in target.split(',')]
    

    test_str = completion.strip()

    if not case_sensitive:
        test_str = test_str.lower()
        target_list = [t.lower() for t in target_list]
    
    return any([sub_target in test_str for sub_target in target_list])

def calculate_accuracy(generated_answers_file: str, ground_truth_file: str):
    """
    主函数，用于加载数据、比较答案并计算最终的准确率。
    """
    print(f"--- 开始评估 ---")
    print(f"模型答案文件: {generated_answers_file}")
    print(f"标准答案文件: {ground_truth_file}")

    if not os.path.exists(generated_answers_file):
        print(f"错误: 找不到模型生成的答案文件 '{generated_answers_file}'")
        return

    with open(generated_answers_file, 'r', encoding='utf-8') as f:
        generated_answers = [line.strip() for line in f]

    if not os.path.exists(ground_truth_file):
        print(f"错误: 找不到标准答案文件 '{ground_truth_file}'")
        return
        
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
        ground_truth_answers = [item['completion'] for item in ground_truth_data]

    if len(generated_answers) != len(ground_truth_answers):
        print("\n警告: 模型生成的答案数量 ({}) 与标准答案数量 ({}) 不匹配!".format(
            len(generated_answers), len(ground_truth_answers)
        ))
        print("将只评估两个文件中共有的部分。")
        min_len = min(len(generated_answers), len(ground_truth_answers))
        generated_answers = generated_answers[:min_len]
        ground_truth_answers = ground_truth_answers[:min_len]

    if not generated_answers:
        print("错误: 未能加载任何答案进行评估。")
        return
-
    correct_count = 0
    total_count = len(generated_answers)
    
    print("\n--- 开始逐一比较答案 ---")
    for i, (gen_ans, truth_ans) in enumerate(zip(generated_answers, ground_truth_answers)):
        is_correct = evaluate_completion(gen_ans, truth_ans)
        if is_correct:
            correct_count += 1
        if i < 5:
            print(f"样本 {i+1}:")
            print(f"  模型答案: '{gen_ans}'")
            print(f"  标准答案: '{truth_ans}'")
            print(f"  评估结果: {'正确' if is_correct else '错误'}")
            print("-" * 20)

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    print("\n--- 评估完成 ---")
    print(f"总计评估样本数: {total_count}")
    print(f"回答正确样本数: {correct_count}")
    print(f"最终准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    # 1. 请将这里的第一个文件名修改为您用 LLaDA 模型生成的答案文件名。
    #    例如，如果您处理的是 qa_positive_negative_questions.txt，
    #    那么生成的文件名可能是 "answers_qa_positive_negative_questions.txt"
    GENERATED_ANSWERS_FILE = "answers_qa_positive_positive_prompt_questions.txt" 
    
    # 2. 请将这里的第二个文件名修改为您上传的、包含标准答案的原始 JSON 文件名。
    GROUND_TRUTH_JSON_FILE = "qa_positive_positive_ground_truth.json"


    calculate_accuracy(GENERATED_ANSWERS_FILE, GROUND_TRUTH_JSON_FILE)