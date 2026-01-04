import json

def process_and_save_data(input_filename="ar_train_dataset.json"):
    """
    Reads a JSON dataset, processes four different QA prompt types into the
    LLaDA SFT format, and saves each type to a separate .txt file.

    Args:
        input_filename (str): The name of the input JSON file.
    """
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{input_filename}'.")
        return

    processed_data = {
        "qa_positive_positive_prompt": [],
        "qa_positive_negative_prompt": [],
        "qa_negative_positive_prompt": [],
        "qa_negative_negative_prompt": []
    }

    prompt_keys = list(processed_data.keys())

    for item in data:
        for key in prompt_keys:
            if key in item:
                prompt_text = item[key]

                parts = prompt_text.split('?', 1)
                if len(parts) == 2:
                    question = parts[0].strip() + '?'
                    answer = parts[1].strip()

                    formatted_string = (
                        f"<BOS><start_id>user<end_id>\\n{question}<eot_id>"
                        f"<start_id>assistant<end_id>\\n{answer}<EOS>"
                    )
                    processed_data[key].append(formatted_string)


    for key, lines in processed_data.items():
        output_filename = f"{key}.txt"
        with open(output_filename, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        print(f"Successfully created {output_filename} with {len(lines)} lines.")


process_and_save_data()