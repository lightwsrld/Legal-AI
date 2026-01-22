import asyncio
import re
import pandas as pd
from tqdm import tqdm
from prompt_rewrite import * 
from openai import AsyncOpenAI


API_BASE = "http://localhost:8000/v1"  
MODEL = "Qwen/Qwen3-235B-A22B-Thinking-2507"  

client = AsyncOpenAI(
    base_url=API_BASE,
    api_key="EMPTY",
    timeout=1800.0,
)

df = pd.read_csv("/./dataset/raw/korean_law_open_data_precedents/train_with_queries.csv")

def build_mcqa_prompt_single(input_query: str, precedent: str, judge: str) -> str:
    return MCQA_PROMPT_SINGLE.format(
        input_query=input_query,
        precedent=precedent,
        judge=judge,
    )

def build_mcqa_prompt_double(input_query: str, precedent: str, judge_1: str, judge_2: str) -> str:
    return MCQA_PROMPT_DOUBLE.format(
        input_query=input_query,
        precedent=precedent,
        judge_1=judge_1,
        judge_2=judge_2,
    )


def is_empty(x):
    """판결요지_2가 비었는지 판단 (NaN, None, 공백 문자열 등)"""
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    if isinstance(x, str) and not x.strip():
        return True
    return False


def parse_numbered_items(text: str):
    """
    </think> 뒤에 나오는 '1. ~ 2. ~ 3. ~ 4. ~' 블록을 파싱해서
    (item1, item2, item3, item4) 반환.
    실패하면 (None, None, None, None) 반환.
    """
    if not text:
        return None, None, None, None

    end_tag = "</think>"
    idx = text.rfind(end_tag)
    if idx != -1:
        text = text[idx + len(end_tag):]
    text = text.strip()

    pattern = r'1\.\s*(.*?)\s*2\.\s*(.*?)\s*3\.\s*(.*?)\s*4\.\s*(.*)'
    match = re.search(pattern, text, re.S)

    if not match:
        print("파싱 실패 텍스트:")
        print(text)
        print("=" * 40)
        return None, None, None, None

    item1, item2, item3, item4 = match.groups()
    return item1.strip(), item2.strip(), item3.strip(), item4.strip()

async def call_model_single(row_idx: int = 0):
    row = df.iloc[row_idx]
    input_query = row["query"]
    precedent = row["전문"]

    judge_1 = row.get("판결요지_1", None)
    judge_2 = row.get("판결요지_2", None)

    # 판결요지_2 존재 여부에 따라 프롬프트 선택
    if is_empty(judge_2):
        # 판결요지 1개인 경우
        mcqa_prompt = build_mcqa_prompt_single(input_query, precedent, judge_1)
    else:
        # 판결요지 2개인 경우
        mcqa_prompt = build_mcqa_prompt_double(input_query, precedent, judge_1, judge_2)

    messages = [
        {"role": "system", "content": "너는 한국어로 답하는 법률 추론 어시스턴트이다."},
        {"role": "user", "content": mcqa_prompt},
    ]

    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=16384,
        temperature=0.6,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    msg = response.choices[0].message
    content = msg.content or getattr(msg, "reasoning_content", "")

    print("===== 모델 출력 =====")
    print(content)

    wrong1, wrong2, wrong3, wrong4 = parse_numbered_items(content)

    return row_idx, content, wrong1, wrong2, wrong3, wrong4

async def call_model_parallel(row_indices, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_single(idx):
        async with semaphore:
            try:
                return await call_model_single(idx)
            except Exception as e:
                print(f"오류 {idx}: {e}")
                return None

    results = []
    tasks = [asyncio.create_task(run_single(idx)) for idx in row_indices]

    pbar = tqdm(total=len(tasks), desc="MCQA inference", ncols=100)

    for coro in asyncio.as_completed(tasks):
        res = await coro
        results.append(res)
        pbar.update(1)

    pbar.close()
    return results


if __name__ == "__main__":
    row_indices = [
        idx
        for idx, row in df.iterrows()
        if not is_empty(row.get("판결요지_2", None))
    ]

    print(f"전체 행 수: {len(df)}")
    print(f"'판결요지_2' 있는 행 수: {len(row_indices)}")

    results = asyncio.run(call_model_parallel(row_indices, max_concurrent=10))

    df["mcqa_raw"] = None
    df["answer1"] = None
    df["answer2"] = None
    df["answer3"] = None
    df["answer4"] = None

    save_every = 50
    out_path = "./dataset/rlvr/legal_ko/precedent.csv"

    processed = 0

    for res in results:
        if res is None:
            processed += 1
            continue

        row_idx, raw_output, w1, w2, w3, w4 = res
        df.loc[row_idx, "mcqa_raw"] = raw_output
        df.loc[row_idx, "answer1"] = w1
        df.loc[row_idx, "answer2"] = w2
        df.loc[row_idx, "answer3"] = w3
        df.loc[row_idx, "answer4"] = w4

        processed += 1

        if processed % save_every == 0:
            print(f"체크포인트 저장: {processed}개 완료 → {out_path}")
            df.to_csv(out_path, index=False, encoding="utf-8-sig")

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("최종 저장 완료 →", out_path)


