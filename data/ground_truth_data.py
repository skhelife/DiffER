import json
import os


def extract_question_and_answer(prompt_text: str) -> (str, str):

    parts = prompt_text.split('?', 1)
    
    if len(parts) == 2:
        question = parts[0].strip() + "?"
        
        answer = parts[1].strip()
        if answer.endswith('.'):
            answer = answer[:-1].strip()
            
        return question, answer
    
    return None, None


def create_evaluation_files(input_file: str):
    """
    Read the original data file, extract all question-answer pairs of the four
    specified types, and create a new JSON file for each type.
    """
    print(f"--- Starting to process file: {input_file} ---")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found '{input_file}'")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing JSON file: {e}")
        return

    data_lists = {
        "pp": [],
        "pn": [],
        "nn": [],
        "np": []
    }

    prompt_keys = {
        "pp": "qa_positive_positive_prompt",
        "pn": "qa_positive_negative_prompt",
        "nn": "qa_negative_negative_prompt",
        "np": "qa_negative_positive_prompt"
    }
    
    print("Extracting all question-answer pairs of the four specified types...")
    
    for item in original_data:
        for key_abbr, prompt_key in prompt_keys.items():
            if prompt_key in item:
                full_prompt = item[prompt_key]
                question, answer = extract_question_and_answer(full_prompt)
                
                if question is not None and answer is not None:
                    data_lists[key_abbr].append({
                        "prompt": question + " A:",
                        "completion": answer
                    })
                        
    print("\n--- Starting to save files ---")

    output_files_data = {
        "qa_positive_positive_ground_truth.json": data_lists["pp"],
        "qa_positive_negative_ground_truth.json": data_lists["pn"],
        "qa_negative_negative_ground_truth.json": data_lists["nn"],
        "qa_negative_positive_ground_truth.json": data_lists["np"]
    }

    for filename, data in output_files_data.items():
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully created file: {filename}, containing {len(data)} data entries.")


if __name__ == "__main__":
    INPUT_JSON_FILE = "ar_train_dataset.json"
    create_evaluation_files(INPUT_JSON_FILE)
