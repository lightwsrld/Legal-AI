import os
import json
import csv
import time
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from threading import Lock
import inspect
from langchain_openai import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from prompt import *

os.environ['OPENAI_API_KEY'] = ""

JSONL_PATH = "./dataset/raw/law_qa_dataset/law_qa_dataset.jsonl"

csv_lock = Lock()

class LawAdvisorLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)

    def generate_judge_label(self, input, answer):
        response_schemas = [
            ResponseSchema(
                name="answer", description="The answer to the user's question."
            )
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_template(
            QA_PROMPT,
            partial_variables={"format_instructions": format_instructions},
        )

        chain = prompt | self.llm | output_parser

        chain = chain.with_config(
            {
                "run_name": inspect.currentframe().f_code.co_name,
                "tags": [self.llm.model_name],
            }
        )

        result = chain.invoke({"input_str": input, "answer": answer})

        return result["answer"]

def load_jsonl_lines(jsonl_path):
    """
    JSONL 파일을 한 줄씩 읽어서 리스트로 반환
    """
    lines = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 빈 줄 건너뛰기
                lines.append(line)
    return lines

def process_single_jsonl_line(line_data, law_llm):
    """
    JSONL의 한 줄(dict)을 처리하는 함수
    """
    try:
        case_data = line_data

        json_str = json.dumps(case_data, ensure_ascii=False)

        question = case_data.get("question", "")  
        answer = case_data.get("answer", "")  

        label = law_llm.generate_judge_label(question, answer)

        return {
            "content": json_str,
            "answer": answer,
            "label": label,
            "error": "",
            "success": True
        }
    except Exception as e:
        return {
            "content": json_str if 'json_str' in locals() else "",
            "answer": answer if 'answer' in locals() else "",
            "label": "",
            "error": repr(e),
            "success": False
        }

def parallel_process_with_intermediate_save(jsonl_lines, law_llm, out_csv, max_workers=10, save_interval=100):
    results = [None] * len(jsonl_lines)
    fieldnames = ["content", "answer", "label", "error"]

    append_mode = out_csv.exists()
    if not append_mode:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    buffer = []
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_jsonl_line, json.loads(line), law_llm): i
            for i, line in enumerate(jsonl_lines)
        }

        with tqdm(total=len(jsonl_lines), desc="Processing JSONL lines") as pbar:
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                result = future.result()
                results[index] = result

                buffer.append(result)
                completed_count += 1
                pbar.update(1)

                if len(buffer) >= save_interval or completed_count == len(jsonl_lines):
                    with csv_lock:
                        with open(out_csv, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            for buffered_result in buffer:
                                row_data = {
                                    "content": buffered_result["content"],
                                    "answer": buffered_result["answer"],
                                    "label": buffered_result["label"],
                                    "error": buffered_result["error"]
                                }
                                writer.writerow(row_data)
                            f.flush()
                            os.fsync(f.fileno())

                    success_in_buffer = sum(1 for r in buffer if r["success"])
                    fail_in_buffer = len(buffer) - success_in_buffer

                    pbar.set_postfix({
                        'saved': f"{completed_count}/{len(jsonl_lines)}",
                        'buffer_success': success_in_buffer,
                        'buffer_fail': fail_in_buffer
                    })

                    buffer = []

    return results

def main(law_llm, max_workers=10, do_parallel=True, save_interval=100, test_count=None):
    """
    메인 처리 함수
    save_interval: 중간 저장 간격 (기본값 100개)
    test_count: 테스트용으로 처리할 라인 개수 (None이면 전체 처리)
    """

    print(f"Loading JSONL file: {JSONL_PATH}")
    all_jsonl_lines = load_jsonl_lines(JSONL_PATH)


    if test_count is not None:
        all_jsonl_lines = all_jsonl_lines[:test_count]
        print(f"TEST MODE: Processing only {test_count} lines")

    total_count = len(all_jsonl_lines)

    if total_count == 0:
        print(f"No lines found in {JSONL_PATH}")
        return

    print(f"Total JSONL lines to process: {total_count}")
    print(f"Parallel processing: {do_parallel}")
    print(f"Intermediate save every: {save_interval} lines")

    jsonl_parent = Path(JSONL_PATH).parent
    out_csv = jsonl_parent / "../data/law_qa_cot_2.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if do_parallel:
        print(f"Starting parallel processing with {max_workers} workers...")
        results = parallel_process_with_intermediate_save(
            all_jsonl_lines, law_llm, out_csv, max_workers, save_interval
        )
    else:
        print("Starting sequential processing...")
        results = []
        fieldnames = ["content", "answer", "label", "error"]


        append_mode = out_csv.exists()
        if not append_mode:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        buffer = []
        for i, line in enumerate(tqdm(all_jsonl_lines, desc="Processing JSONL lines")):
            line_data = json.loads(line)
            result = process_single_jsonl_line(line_data, law_llm)
            results.append(result)
            buffer.append(result)

            if len(buffer) >= save_interval or i == len(all_jsonl_lines) - 1:
                with open(out_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    for buffered_result in buffer:
                        row_data = {
                            "content": buffered_result["content"],
                            "answer": buffered_result["answer"],
                            "label": buffered_result["label"],
                            "error": buffered_result["error"]
                        }
                        writer.writerow(row_data)
                    f.flush()
                    os.fsync(f.fileno())
                buffer = []

    elapsed_time = time.time() - start_time
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count

    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Lines processed: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Results saved to: {out_csv}")

# 실행
if __name__ == "__main__":

    law_llm = LawAdvisorLLM(model_name="gpt-5")

    main(law_llm, max_workers=10, do_parallel=True, save_interval=1000)
