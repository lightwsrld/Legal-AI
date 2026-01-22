import asyncio
import re
import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI


API_BASE = "http://localhost:8005/v1"
MODEL = "qwen3-8b"  

client = AsyncOpenAI(
    base_url=API_BASE,
    api_key="EMPTY",  
    timeout=1800.0,
)

CSV_PATH = "./dataset/rlvr/legal_ko/precedent.csv"
df = pd.read_csv(CSV_PATH)


def format_question_with_answers(row):
    """
    question과 answer1~5를 결합해서
    모델에 던질 MCQA 본문을 만든다.
    """
    answers = []
    for i in range(1, 6):
        ans = row.get(f"answer{i}", "")
        if pd.notna(ans) and str(ans).strip():
            answers.append(f"{i}. {ans}")

    prompt_text = row["question"] + "\n\n" + "\n".join(answers)
    return prompt_text


def build_final_prompt(prompt_text: str) -> str:
    """
    <think> / <answer> 태그를 포함한 최종 유저 프롬프트.
    answer는 1~5 중 하나의 숫자(정답 선택지 번호)만 출력하도록 강제.
    """
    return f"""
    A conversation between User and Assistant. The user asks a question, and the assistant solves it.
    The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e.,
    <think> reasoning process here </think> <answer> answer here </answer>.
    Answer must be a single number between 1 and 5.

User:
{prompt_text}

Assistant:
""".strip()


# <answer>...</answer> 안의 1~5 숫자만 뽑는 정규식
ANSWER_RE = re.compile(r"<answer>\s*([1-5])\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str):
    """
    모델 출력 전체에서 <answer>...</answer> 중 숫자만 추출.
    실패하면 None 반환.
    """
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


async def call_model_single(row_idx: int):
    """
    df의 특정 row 하나에 대해:
    - 프롬프트 생성
    - vLLM 서버에 질의
    - 출력에서 <answer> 숫자 파싱
    """
    row = df.iloc[row_idx]
    mcqa_body = format_question_with_answers(row)
    user_prompt = build_final_prompt(mcqa_body)

    messages = [
        {
            "role": "system",
            "content": "You are a careful Korean legal assistant who always provides detailed reasoning before giving a final answer. Answer in Korean."
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

        msg = response.choices[0].message

        content = msg.content or getattr(msg, "reasoning_content", "")

        pred = extract_answer(content)
        gold_raw = row.get("solution", None)

        try:
            gold = int(str(gold_raw).strip())
        except Exception:
            gold = None

        correct = int(pred == gold) if (pred is not None and gold is not None) else 0

        return {
            "row_idx": row_idx,
            "raw_output": content,
            "pred": pred,
            "gold": gold,
            "correct": correct,
        }

    except Exception as e:
        print(f"❌ 오류 row {row_idx}: {e}")
        return {
            "row_idx": row_idx,
            "raw_output": None,
            "pred": None,
            "gold": None,
            "correct": 0,
        }


async def call_model_parallel(row_indices, max_concurrent=10):
    """
    여러 row index에 대해 비동기 병렬로 call_model_single 실행.
    max_concurrent: 동시에 날릴 최대 요청 수(세마포어로 제한).
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_single(idx):
        async with semaphore:
            return await call_model_single(idx)

    tasks = [asyncio.create_task(run_single(idx)) for idx in row_indices]

    results = []
    pbar = tqdm(total=len(tasks), desc="Qwen3 MCQA inference", ncols=100)

    for coro in asyncio.as_completed(tasks):
        res = await coro
        results.append(res)
        pbar.update(1)

    pbar.close()
    return results


if __name__ == "__main__":

    row_indices = list(df.index)

    results = asyncio.run(call_model_parallel(row_indices, max_concurrent=10))

    # 결과 저장용 컬럼 준비
    if "model_raw" not in df.columns:
        df["model_raw"] = None
    if "model_pred" not in df.columns:
        df["model_pred"] = None
    if "correct" not in df.columns:
        df["correct"] = None


    for res in results:
        idx = res["row_idx"]
        df.loc[idx, "model_raw"] = res["raw_output"]
        df.loc[idx, "model_pred"] = res["pred"]
        df.loc[idx, "correct"] = res["correct"]

    valid_mask = df["model_pred"].notna() & df["solution"].notna()
    if valid_mask.sum() > 0:
        acc = df.loc[valid_mask, "correct"].mean()
        print(f"샘플 {valid_mask.sum()}개 기준 정답률: {acc:.4f}")
    else:
        print("유효한 pred/solution 쌍이 없습니다.")

    OUT_PATH = "./result/qwen8b_result_leet.csv"
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print("결과 저장 완료 →", OUT_PATH)