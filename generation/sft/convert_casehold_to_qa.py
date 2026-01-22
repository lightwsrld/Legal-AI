#!/usr/bin/env python3
"""
Convert CaseHOLD data format to question-answer format
Input format: {context, endings, reasoning, label}
Output format: {question, answer}
"""

import json
import sys

def convert_line(line):
    """Convert a single line from CaseHOLD format to Q&A format"""
    try:
        data = json.loads(line.strip())

        # Combine context and endings to create question
        question_parts = [data['context']]

        # Add endings as numbered options
        if 'endings' in data and data['endings']:
            question_parts.append("\n\nOptions:")
            for i, ending in enumerate(data['endings']):
                question_parts.append(f"{i}: {ending}")

        question = '\n'.join(question_parts)

        # Use reasoning as answer
        answer = data.get('reasoning', '')

        # Add label information if present
        if 'label' in data:
            answer += f"\n\nFinal Answer: {data['label']}"

        return {
            'question': question,
            'answer': answer
        }
    except Exception as e:
        print(f"Error processing line: {e}", file=sys.stderr)
        return None

def main():
    input_file = './dataset/raw/CaseHOLD-R/data/train-00000-of-00001.jsonl'
    output_file = './dataset/sft/legal_en/CaseHOLD_converted_QA.jsonl'

    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")

    converted_count = 0
    error_count = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:

            for line_num, line in enumerate(infile, 1):
                if not line.strip():
                    continue

                converted_data = convert_line(line)

                if converted_data:
                    outfile.write(json.dumps(converted_data, ensure_ascii=False) + '\n')
                    converted_count += 1

                    if converted_count % 1000 == 0:
                        print(f"Processed {converted_count} entries...")
                else:
                    error_count += 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"\nConversion complete!")
    print(f"Successfully converted: {converted_count} entries")
    print(f"Errors: {error_count} entries")

    return 0

if __name__ == '__main__':
    sys.exit(main())