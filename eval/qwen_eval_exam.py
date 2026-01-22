# 파일명: run_qwen_csv_output_with_parsing_exam_sample.py

import asyncio
import pandas as pd
import os
import re
import glob
from tqdm import tqdm
from vLLM_SDK_API import VLLM_API_V1 # 이전 단계에서 생성한 VLLM 클라이언트 클래스
from parsing import parse_model_response, load_prompts_with_indices
from analyze_issues import analyze_single_file


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

    answer_pattern = r'<Answer>\s*(\d+)[\s\S]*?</Answer>'
    matches = re.findall(answer_pattern, combined, re.IGNORECASE)
    if matches:
        return matches[-1]

    return None

# 1. 모델 및 서버 설정
MODEL_PATH = "./.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
BASE_URL = "http://localhost:8005/v1"
SERVED_MODEL_NAME = "qwen3-32b"

SYSTEM_PROMPT = "문제 풀이를 마친 후, 최종 정답을 다음 형식으로 작성해 주세요. <Answer> 정답 </Answer>"
MAX_TOKENS = 8192
TEMPERATURE = 0.0
TOP_P = 1.0
TOKENIZER_ARGS = {"enable_thinking": True}
# TOKENIZER_ARGS = {"enable_thinking": False}

# 필요 시 penalty 파라미터 추가
# EXTRA_BODY_ARGS = {"presence_penalty": 1.5}

# 4. 병렬 처리 설정
SEM = asyncio.Semaphore(40)


async def bounded_generate(idx_prompt: tuple, api_client: VLLM_API_V1):
    idx, prompt = idx_prompt
    async with SEM:
        # 설정된 파라미터를 사용하여 API 클라이언트 호출
        result = await api_client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            tokenizer_args=TOKENIZER_ARGS
            # EXTRA_BODY_ARGS가 설정된 경우 아래와 같이 추가
            # extra_body_args=EXTRA_BODY_ARGS
        )
        return idx, result

async def process_single_csv(csv_file_path: str, api_client: VLLM_API_V1):
    """단일 CSV 파일을 처리하고, 파싱된 결과를 다시 해당 파일에 저장합니다."""
    print(f"Processing {csv_file_path}...")

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        return

    df_filtered = df.copy()
    df_filtered.reset_index(drop=True, inplace=True)

    if df_filtered.empty:
        print(f"No matching rows found in {csv_file_path}.")
        return

    for col in ['think', 'assistant', 'extracted_answer', 'is_correct']:
        if col not in df_filtered.columns:
            df_filtered[col] = ''

    prompts_with_indices = []
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing questions"):
        prompt_parts = []

        if pd.notna(row.get("question")):
            prompt_parts.append(f"[question]\n{row['question']}")

        options = []
        for i in range(1, 6):
            col = f'answer{i}'
            if col in row and pd.notna(row.get(col)) and str(row[col]).strip():
                options.append(f"{i}. {row[col]}")

        if options:
            prompt_parts.append(f"[options]\n" + "\n".join(options))

        full_prompt = "\n\n".join(prompt_parts)
        prompts_with_indices.append((idx, full_prompt))

    tasks = [bounded_generate(ip, api_client) for ip in prompts_with_indices]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
        result = await coro
        results.append(result)

    for idx, raw_result in results:
        parsed = parse_model_response(raw_result)
        df_filtered.loc[idx, 'think'] = parsed['thinking']
        df_filtered.loc[idx, 'assistant'] = parsed['assistant']

    total = 0
    correct = 0
    failed = 0
    failed_indices = []

    for idx, row in df_filtered.iterrows():
        solution = str(row.get('solution', '')).strip()
        if not solution:
            solution = str(row.get('answer', '')).strip()
        if not solution:
            solution = str(row.get('correct_answer', '')).strip()

        think_content = row.get('think', '')
        assistant_content = row.get('assistant', '')

        extracted = extract_answer_from_output(think_content, assistant_content)
        df_filtered.loc[idx, 'extracted_answer'] = extracted

        if extracted is None:
            failed += 1
            df_filtered.loc[idx, 'is_correct'] = None
            failed_indices.append(idx)
        else:
            total += 1
            if solution == extracted:
                correct += 1
                df_filtered.loc[idx, 'is_correct'] = True
            else:
                df_filtered.loc[idx, 'is_correct'] = False

    print("\n" + "=" * 80)
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"{'전체':<30} {total:>8} {correct:>8} {accuracy:>9.2f}%")
    print(f"{'파싱 실패':<30} {failed:>8}")
    print("=" * 80)

    base_name = os.path.basename(csv_file_path).replace(".csv", "")
    output_dir = "./eval/result"
    os.makedirs(output_dir, exist_ok=True)

    thinking_label = "thinking" if TOKENIZER_ARGS.get("enable_thinking", False) else "nonthinking"

    output_path = os.path.join(
        output_dir,
        f"{base_name}_{SERVED_MODEL_NAME}_{thinking_label}.csv"
    )
    df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"Finished processing. Results saved to {output_path}")
    print(f"Total prompts processed: {len(results)}")

    print(f"Analyzing results for {os.path.basename(csv_file_path)}...")
    result = analyze_single_file(output_path)
    if result and result[0] is not None:
        problem_count, total_rows, problem_indices = result
        if problem_count == 0:
            print(f"No issues found - all {total_rows} rows processed successfully!")
        else:
            print(f"Found {problem_count} problematic rows out of {total_rows} total rows")

async def main():
    """
    데이터 디렉토리의 모든 CSV 파일을 찾아 병렬로 처리하고,
    파싱된 결과를 다시 원본 CSV 파일에 저장
    """
    csv_files = ["./data/caselaw_test.csv"]
    # csv_files = ["./data/exam_sample_60.csv"]
    

    print("--- Creating VLLM API Client instance ---")
    api_client = VLLM_API_V1(
        model_path=MODEL_PATH,
        base_url=BASE_URL,
        served_model_name=SERVED_MODEL_NAME
    )
    print("-------------------------------------------\n")

    processing_tasks = [process_single_csv(csv_file, api_client) for csv_file in csv_files]
    await asyncio.gather(*processing_tasks)

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
