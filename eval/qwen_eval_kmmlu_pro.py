# 파일명: run_qwen_kmmlu_pro.py

import asyncio
import json
import os
import re
from tqdm import tqdm
import sys
sys.path.append("./Run_vLLM")
from vLLM_SDK_API import VLLM_API_V1
from parsing import parse_model_response

MODEL_PATH = "./.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
BASE_URL = "http://localhost:8005/v1"
SERVED_MODEL_NAME = "qwen3-32b"

INPUT_FILE = ".data/kmmlu_pro.jsonl"

TARGET_LICENSES = ["변호사", "법무사"]

SYSTEM_PROMPT = "문제 풀이를 마친 후, 최종 정답을 다음 형식으로 작성해 주세요. <Answer> 정답 </Answer> 정답은 선택지 번호 중 하나여야 합니다."
MAX_TOKENS = 8192
TEMPERATURE = 0.0
TOP_P = 1.0
TOKENIZER_ARGS = {"enable_thinking": True}
# TOKENIZER_ARGS = {"enable_thinking": False}

SEM = asyncio.Semaphore(40)


def extract_answer_from_output(think_content, assistant_content):
    """
    think와 assistant에서 <Answer>...</Answer> 태그 안의 숫자를 추출
    - <Answer>3</Answer>
    - <Answer> 3 </Answer>
    - <Answer> 3. ㄱ, ㄴ </Answer> (숫자로 시작하는 경우)
    """
    combined = (think_content or "") + (assistant_content or "")
    if not combined:
        return None

    # <Answer>...</Answer> 태그 안의 내용 추출 (대소문자 무시)
    answer_pattern = r'<Answer>\s*(\d+)[\s\S]*?</Answer>'
    matches = re.findall(answer_pattern, combined, re.IGNORECASE)
    if matches:
        return matches[-1]

    return None


async def bounded_generate(idx_prompt: tuple, api_client: VLLM_API_V1):
    idx, prompt = idx_prompt
    async with SEM:
        result = await api_client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            tokenizer_args=TOKENIZER_ARGS
        )
        return idx, result


async def process_jsonl(input_file: str, api_client: VLLM_API_V1):
    """JSONL 파일을 처리하고, 결과를 새 JSONL 파일에 저장합니다."""
    print(f"Processing {input_file}...")

    data_list = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line.strip())
                if row.get('license_name') in TARGET_LICENSES:
                    data_list.append(row)
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        return

    if not data_list:
        print(f"No data found in {input_file} with license_name in {TARGET_LICENSES}.")
        return

    print(f"필터링된 데이터: {len(data_list)}개 (license_name: {TARGET_LICENSES})")

    prompts_with_indices = []
    for idx, row in enumerate(tqdm(data_list, desc="Preparing prompts")):
        prompt_parts = []

        if row.get("question"):
            prompt_parts.append(f"[question]\n{row['question']}")

        options = row.get("options", [])
        if options:
            options_text = []
            for i, opt in enumerate(options, 1):
                options_text.append(f"{i}. {opt}")
            prompt_parts.append(f"[options]\n" + "\n".join(options_text))

        full_prompt = "\n\n".join(prompt_parts)
        prompts_with_indices.append((idx, full_prompt))

    tasks = [bounded_generate(ip, api_client) for ip in prompts_with_indices]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
        result = await coro
        results.append(result)

    for idx, raw_result in results:
        parsed = parse_model_response(raw_result)
        data_list[idx]['think'] = parsed['thinking']
        data_list[idx]['assistant'] = parsed['assistant']
        data_list[idx]['model_output'] = raw_result

    total = 0
    correct = 0
    failed = 0
    subject_stats = {}
    failed_items = []  

    for row in data_list:
        solution = str(row.get('solution', '')).strip()
        think_content = row.get('think', '')
        assistant_content = row.get('assistant', '')
        subject = row.get('subject', 'Unknown')

        if subject not in subject_stats:
            subject_stats[subject] = {'total': 0, 'correct': 0}

        extracted = extract_answer_from_output(think_content, assistant_content)
        row['extracted_answer'] = extracted

        if extracted is None:
            failed += 1
            row['is_correct'] = None
            failed_items.append(row)  
        else:
            total += 1
            subject_stats[subject]['total'] += 1
            if solution == extracted:
                correct += 1
                subject_stats[subject]['correct'] += 1
                row['is_correct'] = True
            else:
                row['is_correct'] = False

    print("\n" + "=" * 80)
    print(f"{'과목명':<30} {'전체':>8} {'정답':>8} {'정확도':>10}")
    print("-" * 80)

    for subject, stats in sorted(subject_stats.items()):
        subj_accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{subject:<30} "
              f"{stats['total']:>8} "
              f"{stats['correct']:>8} "
              f"{subj_accuracy:>9.2f}%")

    print("=" * 80)
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"{'전체':<30} {total:>8} {correct:>8} {accuracy:>9.2f}%")
    print(f"{'파싱 실패':<30} {failed:>8}")
    print("=" * 80)

    thinking_label = "thinking" if TOKENIZER_ARGS.get("enable_thinking", False) else "nonthinking"
    output_dir = os.path.dirname(input_file)
    base_name = os.path.basename(input_file).replace(".jsonl", "")
    output_path = os.path.join(output_dir, f"{base_name}_{SERVED_MODEL_NAME}_{thinking_label}.jsonl")

    with open(output_path, 'w', encoding='utf-8') as f:
        for row in data_list:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f"\n결과가 {output_path}에 저장되었습니다.")
    print(f"총 {len(data_list)}개 처리 완료")

    if failed_items:
        failed_path = os.path.join(output_dir, f"{base_name}_{SERVED_MODEL_NAME}_{thinking_label}.jsonl")
        with open(failed_path, 'w', encoding='utf-8') as f:
            for row in failed_items:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"파싱 실패 항목 {len(failed_items)}개가 {failed_path}에 저장되었습니다.")


async def main():
    print("--- Creating VLLM API Client instance ---")
    api_client = VLLM_API_V1(
        model_path=MODEL_PATH,
        base_url=BASE_URL,
        served_model_name=SERVED_MODEL_NAME
    )
    print("-------------------------------------------\n")

    await process_jsonl(INPUT_FILE, api_client)


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
