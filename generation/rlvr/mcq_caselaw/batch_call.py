import csv
import json
import argparse
from tqdm import tqdm
import random
import os
from api_call_es import system_prompt_v1, system_prompt_v2, system_prompt_v3, system_prompt_v4, system_prompt_v5, json_schema_v1, json_schema_v2, json_schema_v3, call_gpt_with_case, parse_json_or_raise, validate_json_schema

# Define the batch processing function
def process_batch(batch_number, start_row, end_row, system_prompt, json_schema, csv_filename):
    # Load the data from NDJSON
    with open('./generation/rlvr/mcq_caselaw/all_cases.ndjson', 'r', encoding='utf-8') as f:
        lines = f.readlines()[start_row:end_row]
        lines = random.sample(lines, 2872)

    # Prepare the output CSV (append if exists, write header only if empty/nonexistent)
    fieldnames = ['batch_number', 'number', 'caseNm', 'caseNo', 'courtNm', 'abridged_context', 'question', 'answer1', 'answer2', 'answer3', 'answer4', 'answer5', 'correct_answer', 'explanation']
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

        # Process each line
        for i, line in enumerate(tqdm(lines, desc=f'Processing batch {batch_number}')):
            case_data = json.loads(line)

            # Call the API
            raw = call_gpt_with_case(case_data, system_prompt(), json_schema)
            # print("Raw response:", raw)
            if raw:
                try:
                    result_json = parse_json_or_raise(raw)
                    validate_json_schema(result_json)
                except Exception:
                    # 응답 파싱/검증 실패 시 해당 케이스 스킵
                    continue

                for j, item in enumerate(result_json['items']):
                    # 서술형(json_schema_v3, batch 5)에는 선택지가 없으므로 공란 처리
                    has_choices = isinstance(item, dict) and isinstance(item.get('choices'), dict)
                    row = {
                        'number': start_number + 1,
                        'batch_number': batch_number,
                        'caseNm': result_json['meta']['caseNm'],
                        'caseNo': result_json['meta']['caseNo'],
                        'courtNm': result_json['meta']['courtNm'],
                        'abridged_context': item.get('abridged_context', ''),
                        'question': item['question'],
                        'answer1': item['choices']['A'] if has_choices else '',
                        'answer2': item['choices']['B'] if has_choices else '',
                        'answer3': item['choices']['C'] if has_choices else '',
                        'answer4': item['choices']['D'] if has_choices else '',
                        'answer5': item['choices']['E'] if has_choices else '',
                        # 서술형에서는 correct가 서술형 모범답안(문자열)이므로 그대로 기록
                        'correct_answer': item.get('correct', ''),
                        'explanation': item['explanation']
                    }
                    writer.writerow(row)
                    start_number += 1

# Main function to handle argument parsing and batch processing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process batches of questions.')
    parser.add_argument('--batch', type=int, choices=[1, 2, 3, 4, 5], required=True, help='Batch number to process')
    args = parser.parse_args()

    # Define batch parameters
    batch_params = {
        1: {'start_row': 0, 'end_row': 4500, 'system_prompt': system_prompt_v1, 'json_schema': json_schema_v1, 'csv_filename': 'batch1.csv'},
        2: {'start_row': 3001, 'end_row': 6000, 'system_prompt': system_prompt_v2, 'json_schema': json_schema_v1, 'csv_filename': 'batch2.csv'},
        3: {'start_row': 4500, 'end_row': 9000, 'system_prompt': system_prompt_v3, 'json_schema': json_schema_v2, 'csv_filename': 'batch3.csv'},
        3: {'start_row': 4500, 'end_row': 9000, 'system_prompt': system_prompt_v3, 'json_schema': json_schema_v2, 'csv_filename': 'batch3.csv'},
        4: {'start_row': 0, 'end_row': 4500, 'system_prompt': system_prompt_v4, 'json_schema': json_schema_v2, 'csv_filename': 'batch4.csv'},
        5: {'start_row': 4500, 'end_row': 9000, 'system_prompt': system_prompt_v5, 'json_schema': json_schema_v3, 'csv_filename': 'batch5.csv'},
    }

    # Get the parameters for the selected batch
    params = batch_params[args.batch]

    # Process the selected batch
    process_batch(args.batch, params['start_row'], params['end_row'], params['system_prompt'], params['json_schema'], params['csv_filename'])
