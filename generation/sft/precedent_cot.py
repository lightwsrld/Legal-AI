import os
import json
import csv
import time
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from threading import Lock
import pandas as pd
import inspect
import inspect
import unicodedata
from pathlib import Path
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from prompt import *

os.environ['OPENAI_API_KEY'] = ""

class LawAdvisorLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=0.5)
        self.llm_pred_cate = ChatOpenAI(model=model_name, temperature=0.15)

    def generate_judge_label(self, input, jo_text):

        current_class_name = self.__class__.__name__
        current_method_name = inspect.currentframe().f_code.co_name

        response_schemas = [
            ResponseSchema(
                name="answer", description="The answer to the user's question."
            )
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_template(
            JUDGEMENT_PROMPT,
            partial_variables={"format_instructions": format_instructions},
        )

        chain = prompt | self.llm_pred_cate | output_parser

        chain = chain.with_config(
            {
                "run_name": inspect.currentframe().f_code.co_name,
                "tags": [self.llm_pred_cate.model_name],
            }
        )

        result = chain.invoke({"input_str": input, "relate_laword": jo_text})

        return result["answer"]

BASE_DIR = "./dataset/raw/aihub_precedent/01.민사"
YEARS = ["1981~2016"]

EXISTING_CSV = "./dataset/raw/aihub_precedent/01.민사/civil_after_2000.csv"

csv_lock = Lock()

def normalize_name(s: str) -> str:
    """파일명 비교용 정규화: 공백/따옴표 트림, .json 제거, NFC 정규화"""
    s = str(s).strip().strip('"').strip("'")
    if s.lower().endswith(".json"):
        s = s[:-5]
    return unicodedata.normalize("NFC", s)

def iter_json_paths(base_dir: str, year_list: List[str]) -> List[Path]:
    """연도 폴더들 아래의 모든 .json 경로를 재귀 수집"""
    paths = []
    for y in year_list:
        year_dir = Path(base_dir) / y
        if not year_dir.exists():
            continue
        for p in year_dir.rglob("*.json"):
            paths.append(p)
    return paths

def collect_missing_json_paths(base_dir: str, years: List[str], existing_csv_path: str) -> List[Path]:
    """
    CSV에는 없고(=미기록), 폴더에는 존재하는 .json들의 실제 경로 목록을 반환.
    즉, '폴더→CSV 초과(extra)'만 추려냄.
    """

    df = pd.read_csv(existing_csv_path)
    csv_names = set(normalize_name(x) for x in df["filename"].astype(str).tolist())

    stem_to_paths: Dict[str, List[Path]] = {}
    for y in years:
        year_dir = Path(base_dir) / y
        if not year_dir.exists():
            continue
        for p in year_dir.rglob("*.json"):
            stem = normalize_name(p.stem)
            stem_to_paths.setdefault(stem, []).append(p)

    all_stems = set(stem_to_paths.keys())

    extra_stems = sorted(all_stems - csv_names)

    missing_paths: List[Path] = []
    for s in extra_stems:
        missing_paths.extend(stem_to_paths[s])

    return missing_paths

def count_jsons(path):
    count = 0
    for root, dirs, files in os.walk(path):
        count += sum(1 for f in files if f.endswith(".json"))
    return count

def iter_json_paths(base_dir: str, year_list):
    paths = []
    for y in year_list:
        year_dir = Path(base_dir) / y
        if not year_dir.exists():
            continue
        for p in year_dir.rglob("*.json"):
            paths.append(p)
    return paths

def process_single_json(json_path: Path, law_llm: LawAdvisorLLM) -> Dict[str, Any]:
    """단일 JSON 파일 처리 → label 생성 결과 dict 반환"""
    json_str = ""
    try:
        with open(json_path, "r", encoding="utf-8") as jf:
            case_data = json.load(jf)

        json_str = json.dumps(case_data, ensure_ascii=False)
        input_str = json.dumps(case_data, ensure_ascii=False, indent=2)
        jo_text = case_data.get("info", {}).get("relateLaword", "")

        # LLM 호출
        label = law_llm.generate_judge_label(input_str, jo_text)

        return {
            "year": json_path.parts[-2] if len(json_path.parts) >= 2 else "",
            "filename": json_path.stem,
            "content": json_str,
            "label": label,
            "error": "",
            "success": True
        }
    except Exception as e:
        return {
            "year": json_path.parts[-2] if len(json_path.parts) >= 2 else "",
            "filename": json_path.stem,
            "content": json_str,
            "label": "",
            "error": repr(e),
            "success": False
        }

def parallel_process_with_intermediate_save(
    json_paths: List[Path],
    law_llm: LawAdvisorLLM,
    out_csv: Path,
    max_workers: int = 10,
    save_interval: int = 100
) -> List[Dict[str, Any]]:
    """
    대상 JSON 파일들을 병렬 처리하면서 중간 저장.
    """
    results = [None] * len(json_paths)
    fieldnames = ["year", "filename", "content", "label", "error"]

    append_mode = out_csv.exists()
    if not append_mode:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    buffer = []
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_json, path, law_llm): i for i, path in enumerate(json_paths)}

        with tqdm(total=len(json_paths), desc="Processing JSONs (missing-only)") as pbar:
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                result = future.result()
                results[index] = result

                buffer.append(result)
                completed_count += 1
                pbar.update(1)

                if len(buffer) >= save_interval or completed_count == len(json_paths):
                    with csv_lock:
                        with open(out_csv, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            for r in buffer:
                                writer.writerow({
                                    "year": r["year"],
                                    "filename": r["filename"],
                                    "content": r["content"],
                                    "label": r["label"],
                                    "error": r["error"]
                                })
                            f.flush()
                            os.fsync(f.fileno())

                    success_in_buffer = sum(1 for r in buffer if r["success"])
                    fail_in_buffer = len(buffer) - success_in_buffer
                    pbar.set_postfix({
                        'saved': f"{completed_count}/{len(json_paths)}",
                        'buffer_success': success_in_buffer,
                        'buffer_fail': fail_in_buffer
                    })
                    buffer = []

    return results

def main(model_name: str = "gpt-4o", max_workers: int = 10, save_interval: int = 50):

    missing_paths = collect_missing_json_paths(BASE_DIR, YEARS, EXISTING_CSV)
    print(f"Missing-only to process: {len(missing_paths)} files")

    if not missing_paths:
        print("No missing files to process. Everything in CSV already.")
        return

    law_llm = LawAdvisorLLM(model_name=model_name)

    out_csv = Path(EXISTING_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    results = parallel_process_with_intermediate_save(
        missing_paths, law_llm, out_csv, max_workers=max_workers, save_interval=save_interval
    )
    elapsed = time.time() - start

    success_count = sum(1 for r in results if r and r.get("success"))
    fail_count = len(results) - success_count
    print("\n" + "=" * 60)
    print(f"Processing complete (missing-only).")
    print(f"Total: {len(results)} | Success: {success_count} | Fail: {fail_count}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Appended to: {EXISTING_CSV}")

if __name__ == "__main__":
    main(
        model_name="gpt-4o",
        max_workers=10,
        save_interval=50
    )
