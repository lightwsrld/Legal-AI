import os
import re
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict, Any

from google import genai
from google.genai import types

SYSTEM_PROMPT = (
    "You are assisting with tasks from the Langugage Understanding (언어이해) section of the Korean LEET (Law-school Entrance Exam)."
)

FIGURE_REGEX = re.compile(r"<\s*그림\s*(\d*)\s*>")

class DummyUsage:
    def __init__(self):
        self.input_tokens = None
        self.output_tokens = None
        self.output_tokens_details = None

def load_gemini_client() -> genai.Client:
    load_dotenv(dotenv_path="../.env.api_key")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in environment variables.")
    return genai.Client(api_key=api_key)

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
        To explain further: In SUMMARY, briefly explain what steps you'll take to solve the problem.
        In CAPTION, describe the contents of the image(s); if there are multiple images, describe each image, with a particular focus on details that are directly relevant to the question.
        In REASONING, outline a step-by-step thought process you would use to solve the problem based on the image. Based on this, clearly determine whether each option is true or false.
        In CONCLUSION, give the final answer in a direct format, and if it's a multiple choice question, the conclusion should only include the option without repeating what the option is.
        Here's how the format should look:
        <SUMMARY>[Summarize how you will approach the problem and explain the steps you will take to reach the answer.] </SUMMARY>
        <CAPTION>[Provide a detailed description of the image, particularly emphasizing the aspects related to the question.] </CAPTION>
        <REASONING>[Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning.] </REASONING>
        <CONCLUSION>[State the final answer in a clear and direct format.] </CONCLUSION>
        Please apply this format meticulously to analyze the given image and answer the related question, ensuring that the answer matches the standard one perfectly."""
    )

    return "\n\n".join(parts) + "\n\n" + instruction

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

def find_image_path_for_token(year: int, qno: int, token, image_base_path: str):
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
) -> List[Tuple[Optional[int], str]]:

    if "연도" not in row or "문항번호" not in row or pd.isna(row["연도"]) or pd.isna(row["문항번호"]):
        return []

    year = int(row["연도"])
    qno = int(row["문항번호"])

    tokens = extract_figure_tokens_from_row(row)
    results: List[Tuple[Optional[int], str]] = []

    for token in tokens:
        path = find_image_path_for_token(year, qno, token, image_base_path)
        if path and os.path.exists(path):
            results.append((token, path))

    return results

def call_gemini_with_images(
    client: genai.Client,
    model_name: str,
    prompt: str,
    images: List[Tuple[Optional[int], str]],
):

    contents = []

    contents.append(types.Part(text=prompt))

    for token, path in images:
        try:
            with open(path, "rb") as f:
                image_bytes = f.read()

            if path.lower().endswith('.png'):
                mime_type = "image/png"
            elif path.lower().endswith(('.jpg', '.jpeg')):
                mime_type = "image/jpeg"
            else:
                mime_type = "image/jpeg"  

            contents.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=image_bytes
                    )
                )
            )
        except Exception as e:
            print(f"Warning: Could not load image {path}: {e}")
            continue

    gen_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=gen_config,
    )

    usage = DummyUsage()

    return response.text, usage

def run_single_row(
    csv_path: str,
    image_base_path: str,
    row_index: int,
    model_name: str = "gemini-3-flash-preview",
    dotenv_path: str = "../.env.api_key",
) -> Dict[str, Any]:

    df = pd.read_csv(csv_path)
    row = df.iloc[row_index]

    prompt = build_prompt(row)
    images = collect_images_in_text_order(row, image_base_path)

    print("ATTACHING:", [(t, os.path.basename(p)) for (t, p) in images])
    print("DEBUG len(images) =", len(images))

    client = load_gemini_client()
    output_text, usage = call_gemini_with_images(client, model_name, prompt, images)

    result = {
        "row_index": row_index,
        "연도": int(row["연도"]) if "연도" in row and pd.notna(row["연도"]) else None,
        "문항번호": int(row["문항번호"]) if "문항번호" in row and pd.notna(row["문항번호"]) else None,
        "figure_tokens": extract_figure_tokens_from_row(row),
        "attached_images": [{"token": t, "path": p} for (t, p) in images],
        "input": prompt,
        "response": output_text,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
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
    image_base_path = ".data/images"

    run_single_row(
        csv_path=csv_path,
        image_base_path=image_base_path,
        row_index=93,
        model_name="gemini-3-flash-preview",
    )
