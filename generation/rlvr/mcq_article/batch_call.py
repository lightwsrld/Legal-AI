import csv
import json
import argparse
from tqdm import tqdm
from api_call_v2 import system_prompt_v8, system_prompt_v9, system_prompt_v10, user_prompt, call_gpt_with_prompts, parse_json_or_raise, validate_json_schema
import random
import os

# Define the batch processing function
def process_batch(batch_number, start_row, end_row, system_prompt, csv_filename):
    # Load the data from CSV
    with open('./dataset/raw/lawtimes/lawtimes_all.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)[start_row:end_row]
        rows = random.sample(rows, 3500)

    # Prepare the output CSV (append if exists, write header only if empty/nonexistent)
    fieldnames = ['batch_number', 'number', 'url', 'title', 'abridged_context', 'question', 'answer1', 'answer2', 'answer3', 'answer4', 'answer5', 'correct_answer', 'reason']
    file_exists = os.path.exists(csv_filename)
    file_is_empty = (not file_exists) or os.path.getsize(csv_filename) == 0
    mode = 'a' if file_exists and not file_is_empty else 'w'

    # determine start index to continue numbering
    start_number = 0
    if file_exists and not file_is_empty:
        with open(csv_filename, 'r', encoding='utf-8') as existing:
            # count lines excluding header
            line_count = sum(1 for _ in existing)
            start_number = max(0, line_count - 1)

    with open(csv_filename, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if file_is_empty:
            writer.writeheader()

        # Process each row
        for i, row in enumerate(tqdm(rows, desc=f'Processing batch {batch_number}')):
            url, title, content = row['url'], row['title'], row['content']

            # Call the API
            result = call_gpt_with_prompts(system_prompt(), user_prompt(url, title, content))
            if not result:
                continue
            try:
                result_json = parse_json_or_raise(result)
                validate_json_schema(result_json)
            except Exception:
                # 응답이 유효 JSON이 아니거나 스키마 위반 시 해당 행은 건너뜀
                continue

            for j, item in enumerate(result_json['items']):
                writer.writerow({
                    'batch_number': batch_number,
                    'number': start_number + 1,
                    'url': result_json['meta']['url'],
                    'title': result_json['meta']['title'],
                    'abridged_context': item['abridged_context'],
                    'question': item['question'],
                    'answer1': item['choices']['A'],
                    'answer2': item['choices']['B'],
                    'answer3': item['choices']['C'],
                    'answer4': item['choices']['D'],
                    'answer5': item['choices']['E'],
                    'correct_answer': item['correct'],
                    'reason': item['reason']
                })
                start_number += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process batches of questions.')
    parser.add_argument('--batch', type=int, choices=[1, 2, 3, 4, 5], required=True, help='Batch number to process')
    args = parser.parse_args()

    batch_params = {
        1: {'start_row': 0, 'end_row': 10000, 'system_prompt': system_prompt_v8, 'csv_filename': 'batch1.csv'},
        2: {'start_row': 10000, 'end_row': 19500, 'system_prompt': system_prompt_v9, 'csv_filename': 'batch2.csv'},
        3: {'start_row': 0, 'end_row': 10000, 'system_prompt': system_prompt_v8, 'csv_filename': 'batch3.csv'},
        4: {'start_row': 10000, 'end_row': 19500, 'system_prompt': system_prompt_v9, 'csv_filename': 'batch4.csv'},
        5: {'start_row': 0, 'end_row': 19500, 'system_prompt': system_prompt_v10, 'csv_filename': 'batch5.csv'},
    }

    params = batch_params[args.batch]

    process_batch(args.batch, params['start_row'], params['end_row'], params['system_prompt'], params['csv_filename'])

