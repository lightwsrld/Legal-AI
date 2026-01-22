import argparse
import csv
import json
import os
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = """
You are a bilingual (Korean↔English) legal translator specialized in Korean case law.

Task:
- The question text must remain in Korean (only provided for context).
- Translate ONLY the `solution` (model answer) into precise, natural legal English while preserving every legal nuance, holding, and logical connector.
- Maintain Korean legal concepts and terms as they are (e.g., “민법 제104조” should stay as “Article 104 of the Korean Civil Act,” not adapted to foreign legal systems).
- Do NOT add Korean text, summaries, or extra commentary.
- Output must be valid JSON in the form {"solution": "<english translation>"} with proper escaping. No code blocks.

Schema:
{
  "question": "<Korean question>",
  "answers": {
    "answer1": "",
    "answer2": "",
    "answer3": "",
    "answer4": "",
    "answer5": ""
  },
  "solution": "<translated solution>"
}

""".strip()


def build_user_prompt(row: dict) -> str:
    payload = {
        "question": row.get("question", "") or "",
        "solution": row.get("solution", "") or "",
    }
    return (
        "Question (for context only, keep as-is in Korean):\n"
        f"{payload['question']}\n\n"
        "Translate ONLY the following solution into English and return JSON {\"solution\": \"...\"} with no additional text:\n"
        f"{json.dumps({'solution': payload['solution']}, ensure_ascii=False)}"
    )


def call_translation(row: dict, retries: int = 2) -> dict:
    for attempt in range(retries + 1):
        response = client.chat.completions.create(
            model="gpt-4.1",
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(row)},
            ],
        )
        content = response.choices[0].message.content
        try:
            result = json.loads(content)
            if "solution" not in result:
                raise ValueError("Missing 'solution' key in translation result")
            return result
        except Exception as exc:
            if attempt >= retries:
                raise RuntimeError(f"Failed to parse translation response: {exc}\nRaw: {content}") from exc
            time.sleep(1.5)
    raise RuntimeError("Translation failed after retries")


def translate_csv(input_path: Path, output_path: Path, rate_limit: float = 0.0, batch: int | None = None):
    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = list(csv.DictReader(infile))

        if batch is not None:
            total = len(reader)
            batch_bounds = {
                1: (0, min(1500, total)),
                2: (1500, min(3000, total)),
            }
            if batch not in batch_bounds:
                raise ValueError("batch must be 1 or 2")
            start_idx, end_idx = batch_bounds[batch]
            rows = reader[start_idx:end_idx]
            print(f"[batch {batch}] processing rows {start_idx + 1}~{end_idx} (total {len(rows)})")
        else:
            rows = reader
            print(f"[all] processing entire dataset ({len(rows)} rows)")

        fieldnames = [
            "question",
            "answer1",
            "answer2",
            "answer3",
            "answer4",
            "answer5",
            "solution",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        progress = tqdm(
            rows,
            desc="Translating",
            unit="row",
            total=len(rows),
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for idx, row in enumerate(progress, start=1):
            translation = call_translation(row)
            writer.writerow(
                {
                    "question": row.get("question", ""),
                    "answer1": row.get("answer1", ""),
                    "answer2": row.get("answer2", ""),
                    "answer3": row.get("answer3", ""),
                    "answer4": row.get("answer4", ""),
                    "answer5": row.get("answer5", ""),
                    "solution": translation.get("solution", ""),
                }
            )
            if rate_limit > 0:
                time.sleep(rate_limit)

    print(f"Translation completed: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate caselaw_written.csv rows into English using GPT-4.1"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./generation/rlvr/mcq_caselaw/caselaw_written.csv"),
        help="Path to the source CSV (default: caselaw_written.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the translated CSV (auto-set if batch specified)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls (optional rate limiting)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        choices=[1, 2],
        help="Batch slice to process (1: rows 1-1500, 2: rows 1501-3000). If omitted, process all rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output
    if output_path is None:
        if args.batch == 1:
            output_path = Path("./generation/rlvr/mcq_caselaw/caselaw_written_en_1.csv")
        elif args.batch == 2:
            output_path = Path("./generation/rlvr/mcq_caselaw/caselaw_written_en_2.csv")
        else:
            output_path = Path("./generation/rlvr/mcq_caselaw/caselaw_written_en.csv")

    translate_csv(args.input, output_path, rate_limit=args.sleep, batch=args.batch)


