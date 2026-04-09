import json
import re
import os

def normalize_text(text):
    """
    Normalize text: convert to lowercase, replace punctuation with spaces,
    and remove leading and trailing spaces.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

def extract_subject_from_question(question_text):
    """
    Extract the person's name before 's.
    """
    clean_q = question_text.strip()
    match = re.search(r"^(.*?)'s\b", clean_q)
    if match:
        return match.group(1).strip()
    if "'s" in clean_q:
        return clean_q.split("'s")[0].strip()
    return ""

def evaluate_single_sample(question, ground_truth, model_answer):

    ans_norm = normalize_text(model_answer)
    gt_norm = normalize_text(ground_truth)


    if gt_norm in ans_norm and len(gt_norm) > 0:
        return 0

    ans_tokens = set(t for t in ans_norm.split() if len(t) >= 1)

    subject_name = extract_subject_from_question(question)
    subject_norm = normalize_text(subject_name)
    subj_tokens = [t for t in subject_norm.split() if len(t) >= 2]


    for token in subj_tokens:
        if token in ans_tokens:
            return 1


    gt_tokens = [t for t in gt_norm.split() if len(t) >= 2] 
    
    for token in gt_tokens:
        if token in ans_tokens:
            return 2

    return 3

def save_category_file(output_dir, type_id, type_name, data_list):
    """
    Helper function: write the error list of a specific category to a txt file.
    """
    filename = os.path.join(output_dir, f"type_{type_id}_{type_name}.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"=== Error Type {type_id}: {type_name} ===\n")
        f.write(f"Total Count: {len(data_list)}\n")
        f.write("=" * 50 + "\n\n")
        
        for item in data_list:
            f.write(f"[Line {item['line_id']}]\n")
            f.write(f"Q:   {item['question']}\n")
            f.write(f"GT:  {item['ground_truth']}\n")
            f.write(f"Ans: {item['model_answer']}\n")
            if type_id == 1:
                f.write(f"Subject Detected: {item['extracted_subject']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"Saved: {filename} (contains {len(data_list)} entries)")

def analyze_model_results(questions_file, ground_truth_file, answers_file, output_dir="./analysis_result"):
    print(f"--- Starting error analysis ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]

        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            if isinstance(gt_data, list) and len(gt_data) > 0 and isinstance(gt_data[0], dict):
                ground_truths = [item.get('completion', '').strip() for item in gt_data]
            else:
                ground_truths = [str(item).strip() for item in gt_data]

        with open(answers_file, 'r', encoding='utf-8') as f:
            model_answers = [line.strip() for line in f]

    except Exception as e:
        print(f"Error reading file: {e}")
        return

    min_len = min(len(questions), len(ground_truths), len(model_answers))
    print(f"Number of valid aligned samples: {min_len}")
    if min_len == 0: return

    categorized_data = {
        0: [], # Correct
        1: [], # Repeat Subject
        2: [], # Partial
        3: []  # Wrong
    }

    for i in range(min_len):
        q = questions[i]
        gt = ground_truths[i]
        ans = model_answers[i]

        result_type = evaluate_single_sample(q, gt, ans)
        entry = {
            "line_id": i + 1,  
            "question": q,
            "ground_truth": gt,
            "model_answer": ans,
            "extracted_subject": extract_subject_from_question(q) if result_type == 1 else ""
        }
        categorized_data[result_type].append(entry)

    total = min_len
    stats = {k: len(v) for k, v in categorized_data.items()}
    
    print("\n" + "=" * 50)
    print("Analysis Statistics Report")
    print("=" * 50)
    print(f"Total samples: {total}")
    print(f"Type 0 (Completely Correct): {stats[0]} ({stats[0]/total*100:.2f}%)")
    print(f"Type 1 (Repeated Subject): {stats[1]} ({stats[1]/total*100:.2f}%)")
    print(f"Type 2 (Partially Correct): {stats[2]} ({stats[2]/total*100:.2f}%)")
    print(f"Type 3 (Completely Wrong): {stats[3]} ({stats[3]/total*100:.2f}%)")
    print("=" * 50)

    print("\nSaving categorized files...")
    save_category_file(output_dir, 0, "correct", categorized_data[0])
    save_category_file(output_dir, 1, "repeat_subject", categorized_data[1])
    save_category_file(output_dir, 2, "partial_match", categorized_data[2])
    save_category_file(output_dir, 3, "completely_wrong", categorized_data[3])
    
    print(f"Analysis completed. All results have been saved in the directory: {output_dir}")

if __name__ == "__main__":
    Q_PATH = "qa_negative_negative_questions.txt"
    GT_PATH = "qa_negative_negative_ground_truth.json"
    ANS_PATH = "answers_qa_negative_negative_prompt_questions.txt"
    OUT_DIR = "analysis_output"

    analyze_model_results(Q_PATH, GT_PATH, ANS_PATH, OUT_DIR)
