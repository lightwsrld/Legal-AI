import os
import re
import base64
import mimetypes
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict, Any

os.environ['OPENAI_API_KEY'] = ""

SYSTEM_PROMPT = (
    "You are assisting with tasks from the Langugage Understanding (언어이해) section of the Korean LEET (Law-school Entrance Exam)."
)

FIGURE_REGEX = re.compile(r"<\s*그림\s*(\d*)\s*>")

def load_openai_client() -> OpenAI:
    load_dotenv(dotenv_path="../.env.api_key")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Set it in environment or ../.env.api_key"
        )
    return OpenAI(api_key=api_key)


def build_prompt(row) -> str:
    parts = []
    if "질문" in row and pd.notna(row["질문"]):
        parts.append(f"[질문]\n{row['질문']}")
    if "지문" in row and pd.notna(row["지문"]):
        parts.append(f"[지문]\n{row['지문']}")
    if "보기" in row and pd.notna(row["보기"]):
        parts.append(f"[보기]\n{row['보기']}")
    if "선택지" in row and pd.notna(row["선택지"]):
        parts.append(f"[선택지]\n{row['선택지']}")

    instruction = (
        """I have image(s) and a question that I want you to answer. 
        I need you to strictly follow the format with four specific sections: SUMMARY, CAPTION, REASONING, and CONCLUSION. 
        It is crucial that you adhere to this structure exactly as outlined and that the final answer in the CONCLUSION matches the standard correct answer precisely.
        To explain further: In SUMMARY, briefly explain what steps you’ll take to solve the problem. 
        In CAPTION, describe the contents of the image(s); if there are multiple images, describe each image, with a particular focus on details that are directly relevant to the question.
        In REASONING, outline a step-by-step thought process you would use to solve the problem based on the image. Based on this, clearly determine whether each option is true or false.
        In CONCLUSION, give the final answer in a direct format, and if it’s a multiple choice question, the conclusion should only include the option without repeating what the option is.
        Here’s how the format should look:
        <SUMMARY>[Summarize how you will approach the problem and explain the steps you will take to reach the answer.] </SUMMARY>
        <CAPTION>[Provide a detailed description of the image, particularly emphasizing the aspects related to the question.] </CAPTION>
        <REASONING>[Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning.] </REASONING>
        <CONCLUSION>[State the final answer in a clear and direct format.] 
        Please apply this format meticulously to analyze the given image and answer the related question, ensuring that the answer matches the standard one perfectly."""
    )

    return "\n\n".join(parts) + instruction

def extract_figure_tokens_from_row(row: pd.Series) -> List[Optional[int]]:
    text = ""
    for col in ["질문", "지문", "보기", "선택지"]:
        if col in row and pd.notna(row[col]):
            text += str(row[col]) + " "

    matches = FIGURE_REGEX.findall(text)

    seen = set()
    tokens: List[Optional[int]] = []
    for m in matches:
        token = None if m == "" else int(m)  # <그림>이면 None, <그림 n>이면 n
        if token not in seen:
            seen.add(token)
            tokens.append(token)

    return tokens

def find_image_path(row, image_base_path: str) -> str | None:
    # 파일 규칙: {연도}_{문항번호}.jpeg
    if "연도" not in row or "문항번호" not in row:
        return None
    if pd.isna(row["연도"]) or pd.isna(row["문항번호"]):
        return None

    year = int(row["연도"])
    qno = int(row["문항번호"])

    candidates = [
        os.path.join(image_base_path, f"{year}_{qno}.jpeg"),
        os.path.join(image_base_path, f"{year}_{qno}.jpg"),
        os.path.join(image_base_path, f"{year}_{qno}.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def local_image_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def find_image_path_for_token(year: int, qno: int, token, image_base_path: str):
    import os

    exts = ["jpeg", "jpg", "png", "JPEG", "JPG", "PNG"]

    stems = []
    if token is None:
        stems += [f"{year}_{qno}"]
    else:
        stems += [f"{year}_{qno}_{token}"]
        stems += [f"{year}_{qno}_그림 {token}", f"{year}_{qno}_그림{token}"]

    for stem in stems:
        for ext in exts:
            p = os.path.join(image_base_path, f"{stem}.{ext}")
            if os.path.exists(p):
                return p
    return None


def collect_images_in_text_order(
    row: pd.Series,
    image_base_path: str
) -> List[Tuple[Optional[int], str, str]]:

    if "연도" not in row or "문항번호" not in row or pd.isna(row["연도"]) or pd.isna(row["문항번호"]):
        return []

    year = int(row["연도"])
    qno = int(row["문항번호"])

    tokens = extract_figure_tokens_from_row(row)
    results: List[Tuple[Optional[int], str, str]] = []

    for token in tokens:
        path = find_image_path_for_token(year, qno, token, image_base_path)
        if not path:
            continue
        data_url = local_image_to_data_url(path)
        results.append((token, path, data_url))

    return results

def call_openai_with_images(
    client: OpenAI,
    model_name: str,
    prompt: str,
    images: List[Tuple[Optional[int], str, str]],
):
    content = [{"type": "input_text", "text": prompt}]
    for token, path, data_url in images:
        content.append({"type": "input_image", "image_url": data_url})

    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        reasoning={"effort": "high", "summary": "detailed"},
        tools=[],
        store=True,
    )
    return resp.output_text, resp.usage


def run_single_row(
    csv_path: str,
    image_base_path: str,
    row_index: int,
    model_name: str = "gpt-5",
    dotenv_path: str = "../.env.api_key",
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    row = df.iloc[row_index]

    prompt = build_prompt(row)
    images = collect_images_in_text_order(row, image_base_path)

    print("ATTACHING:", [(t, os.path.basename(p)) for (t, p, _) in images])
    print("DEBUG len(images) =", len(images))

    client = load_openai_client()
    output_text, usage = call_openai_with_images(client, model_name, prompt, images)

    result = {
        "row_index": row_index,
        "연도": int(row["연도"]) if "연도" in row and pd.notna(row["연도"]) else None,
        "문항번호": int(row["문항번호"]) if "문항번호" in row and pd.notna(row["문항번호"]) else None,
        "figure_tokens": extract_figure_tokens_from_row(row),
        "attached_images": [{"token": t, "path": p} for (t, p, _) in images],
        "input": prompt,
        "response": output_text,
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        },
    }

    print(f"row_index: {result['row_index']}")
    print(f"연도: {result['연도']}")
    print(f"문항번호: {result['문항번호']}")
    print(f"tokens: {result['figure_tokens']}")
    print("----- MODEL OUTPUT -----")
    print(output_text)

    return result

if __name__ == "__main__":
    csv_path = "./data/언어이해_final.csv"
    image_base_path = "./data/images"
    
    run_single_row(
        csv_path=csv_path,
        image_base_path=image_base_path,
        row_index=93,
        model_name="gpt-5",
    )