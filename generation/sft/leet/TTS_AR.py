from openai import OpenAI
from tqdm import tqdm
import json
import pandas as pd
import os
import argparse
import concurrent.futures
from threading import Lock
from prompt_AR import *
from prompt_LU import *


api_key = ""

client = OpenAI(api_key=api_key)
save_lock = Lock()

model = "gpt-5"

PROMPT_MAP = {
    "법규범": PROMPT_LEGAL_REASONING,
    "인문": PROMPT_HUMAN_REASONING,
    "사회": PROMPT_SOCIAL_REASONING,
    "과학기술": PROMPT_SCIENCE_REASONING,
    "논증분석": PROMPT_ANALYSIS_REASONING,
    "논쟁/반론": PROMPT_DEBATE1_REASONING,
    "논증평가": PROMPT_SOLVING_REASONING,

    "형식추리": PROMPT_MATH_REASONING,
    "수리추리": PROMPT_MATH_REASONING,
    "논리게임(배열))": PROMPT_MATH_REASONING,
    "논리게임(참거짓)": PROMPT_MATH_REASONING,
}

def ar_social_inference(input_question, prompt_template):
    """단일 질문에 대한 o3 모델 추론"""
    prompt = prompt_template.format(input_question=input_question)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "developer",
                "content": "You are assisting with tasks from the Logical Reasoning & Argumentation (추리논증) section of the Korean LEET (Law-school Entrance Exam)."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        reasoning={
            "effort": "high",
            "summary": "detailed"
        },
        tools=[],
        store=True
    )
    
    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    reasoning_tokens = usage.output_tokens_details.reasoning_tokens if usage.output_tokens_details else "N/A"
    result = response.output_text
    clean_json_str = result.strip()
    clean_json_str = clean_json_str.replace("```json", "").replace("```", "").strip()
    # print(clean_json_str)
    result = json.loads(clean_json_str)
    
    try:
        reasoning = response.output[0].summary[0].text 
    except (AttributeError, IndexError):
        reasoning = "No reasoning provided"
    
    return result, input_tokens, output_tokens, reasoning, reasoning_tokens


def process_single_question(row_data):
    """단일 질문 처리 (병렬 처리용)"""
    i, row = row_data

    # 프롬프트 구성
    prompt_parts = []
    if pd.notna(row.get("질문")):
        prompt_parts.append(f"[질문]\n{row['질문']}")
    if pd.notna(row.get("지문")):
        prompt_parts.append(f"[지문]\n{row['지문']}")
    if pd.notna(row.get("보기")):
        prompt_parts.append(f"[보기]\n{row['보기']}")
    if pd.notna(row.get("선택지")):
        prompt_parts.append(f"[선택지]\n{row['선택지']}")
    question = "\n\n".join(prompt_parts)
    
    question_type = row.get("유형")
    prompt_template = PROMPT_MAP.get(question_type, PROMPT_MAP)

    try:
        result, input_tokens, output_tokens, reasoning_trace, reasoning_tokens = ar_social_inference(question, prompt_template)
        print(f"Result for question {i+1}: {result}")
        
        return {
            'index': i, 
            'response': result['Final answer'],
            'reasoning': json.dumps(result, ensure_ascii=False),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'reasoning_tokens': reasoning_tokens,
            'success': True
        }
    except Exception as e:
        print(f"Error processing question {i+1}: {str(e)}")
        return {
            'index': i,
            'response': f"Error: {str(e)}",
            'reasoning': f"Error: {str(e)}",
            'input_tokens': 0,
            'output_tokens': 0,
            'reasoning_tokens': 0,
            'success': False
        }


def parallel_process_questions(df, df_output, output_file, max_workers=5, batch_save_interval=5):
    """
    병렬로 질문들을 처리
    
    Args:
        df: 입력 데이터프레임
        df_output: 출력 데이터프레임
        max_workers: 최대 워커 스레드 수
        batch_save_interval: 중간 저장 간격
    """
    # 처리할 행들 필터링 (아직 처리하지 않은 것들만)
    rows_to_process = [(i, row) for i, row in df.iterrows() 
                       if df_output.loc[i, "response"] == "before generation"]
    
    if not rows_to_process:
        print("모든 질문이 이미 처리되었습니다.")
        return df_output
    
    results_buffer = []
    processed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 작업을 제출
        futures = {executor.submit(process_single_question, row_data): row_data[0] 
                  for row_data in rows_to_process}
        
        # tqdm 진행바 설정
        with tqdm(total=len(rows_to_process), desc="Processing questions (parallel)") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=60)  # 60초 타임아웃
                    
                    if result['success']:
                        # 결과를 데이터프레임에 저장
                        idx = result['index']
                        df_output.loc[idx, 'response'] = result['response']
                        df_output.loc[idx, 'reasoning'] = result['reasoning']
                        df_output.loc[idx, 'input_tokens'] = result['input_tokens']
                        df_output.loc[idx, 'output_tokens'] = result['output_tokens']
                        df_output.loc[idx, 'reasoning_tokens'] = result['reasoning_tokens']
                        
                        results_buffer.append(result)
                        processed_count += 1
                        
                        # 일정 개수마다 저장
                        if processed_count % batch_save_interval == 0:
                            with save_lock:
                                df_output.to_csv(output_file, index=False)
                                print(f"\n중간 저장 완료: {processed_count}개 처리됨")
                    else:
                        print(f"\n질문 {result['index']+1} 처리 실패")
                    
                    pbar.update(1)
                    
                except concurrent.futures.TimeoutError:
                    original_idx = futures[future]
                    print(f"\n질문 {original_idx+1} 타임아웃")
                    pbar.update(1)
                except Exception as e:
                    original_idx = futures[future]
                    print(f"\n질문 {original_idx+1} 처리 중 에러: {str(e)}")
                    pbar.update(1)
    
    # 최종 저장
    df_output.to_csv(output_file, index=False)
    print(f"\n최종 결과 저장 완료: {output_file}")
    
    return df_output

def build_argparser():  
    parser = argparse.ArgumentParser(description="LEET solver batch inference runner")
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="결과를 저장할 CSV 경로 (예: /home/.../result.csv)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="병렬 워커 수"
    )
    parser.add_argument(
        "--batch_save_interval",
        type=int,
        default=10,
        help="몇 개 처리마다 중간 저장할지"
    )
    return parser

def main():

    args = build_argparser().parse_args() 

    df = pd.read_csv("./dataset/LEET/추리논증_final.csv")
    df_filtered = df[df['유형'].isin(["형식추리", "논리게임(배열))", "논리게임(참거짓)"])]
    # df_filtered = df_filtered[df_filtered['연도'].isin([2025])]
    df_filtered = df_filtered[df_filtered['연도'].isin(list(range(2017,2024)))]
    df_filtered.reset_index(drop=True, inplace=True)
    print(len(df_filtered))

    output_file = args.output_file
    if os.path.exists(output_file):
        df_output = pd.read_csv(output_file)
        df_output = df_output.reindex(df_filtered.index)
        print(f"이전 저장분 {len(df_output)}행 불러옴 — 안 끝난 것만 이어서 처리")
    else:
        df_output = df_filtered.copy()
        df_output['input_tokens'] = 0
        df_output['output_tokens'] = 0
        df_output['response'] = "before generation"
        df_output['reasoning'] = "before generation"
        df_output['reasoning_tokens'] = "before generation"
    
    # 병렬 처리 옵션
    use_parallel = True  # False로 설정하면 기존 순차 처리 방식 사용
    
    if use_parallel:
        df_output = parallel_process_questions(
            df_filtered, 
            df_output, 
            output_file=output_file, 
            max_workers=args.max_workers,
            batch_save_interval=args.batch_save_interval
        )
    else:
        # 기존 순차 처리 방식
        for i, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing questions (sequential)"):
            if df_output.loc[i, "response"] != "before generation":
                continue
            
            prompt_parts = []
            if pd.notna(row.get("question")):
                prompt_parts.append(f"[question]\n{row['question']}")
            if pd.notna(row.get("options")):
                prompt_parts.append(f"[지문]\n{row['options']}")
            question = "\n\n".join(prompt_parts)

            question_type = row.get("유형")
            prompt_template = PROMPT_MAP.get(question_type, PROMPT_MAP)
            
            result, input_tokens, output_tokens, reasoning_trace, reasoning_tokens = ar_social_inference(question, prompt_template)
            print(f"Result for question {i+1}: {result}")
            df_output.loc[i, 'response'] = result['Final answer']
            df_output.loc[i, 'reasoning'] = json.dumps(result, ensure_ascii=False)
            df_output.loc[i, 'input_tokens'] = input_tokens
            df_output.loc[i, 'output_tokens'] = output_tokens
            df_output.loc[i, 'reasoning_tokens'] = reasoning_tokens
            
            save_every = 1
            if (i+1) % save_every == 0:
                df_output.to_csv(output_file, index=False)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
