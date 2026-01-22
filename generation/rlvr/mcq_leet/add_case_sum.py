import inspect
import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
import re
from openai import OpenAI
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['OPENAI_API_KEY'] = ""

def extract_judgement_with_regex(question):
    """정규표현식으로 [판결문] 다음 부분 추출"""
    pattern = r'\[판결문\]\s*(.*)'
    match = re.search(pattern, question, re.DOTALL)
    if match:
        return match.group(1).strip()
    return question

LLM_PROMPT = """
#instruction#
당신은 법률 전문가입니다.

판례 데이터셋 ("Precedent"로 주어짐)을 바탕으로 사건 사실을 요약해 작성해야 합니다.
요약 시 피고인의 행위, 피해자의 상황, 범행 경과, 피해 정도 등 객관적 사실만 작성해야 합니다.
또한 항소 기각, 유죄/무죄 판단, 형량, 양형 사유 등 판결 결과나 재판부의 평가, 법적 결론을 암시하는 문장은 쓰지 마시오.
요약은 5-7줄 사이의 줄글로 작성해야 합니다.

결과는 반드시 한국어로 작성하세요.

Precedent:
"'{precedent}"'

#####
{format_instructions}
"""


class LawAdvisorLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=1)
        self.llm_pred_cate = ChatOpenAI(model=model_name, temperature=1)

    def generate_judge_label(self, precedent):
    
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
            LLM_PROMPT,
            partial_variables={"format_instructions": format_instructions},
        )

        chain = prompt | self.llm_pred_cate | output_parser

        chain = chain.with_config(
            {
                "run_name": inspect.currentframe().f_code.co_name,
                "tags": [self.llm_pred_cate.model_name],
            }
        )

        result = chain.invoke({"precedent": precedent})

        return result['answer']

def process_single_item(idx, line, law_llm):
    """단일 항목 처리 함수"""
    try:
        data = json.loads(line)
        judgement_content = extract_judgement_with_regex(data["question"])

        result = law_llm.generate_judge_label(judgement_content)

        if "query" in data and data["query"]:
            updated_queries = []
            for query_item in data["query"]:
                updated_query = f"{query_item}\n사건 내용: {result}"
                updated_queries.append(updated_query)
            data["query"] = updated_queries

        return {"idx": idx, "data": data, "error": None}

    except Exception as e:
        return {"idx": idx, "data": line, "error": str(e)}

gpt_model = "gpt-5"
law_llm = LawAdvisorLLM(model_name=gpt_model)

input_file = "./dataset/sft/legal_ko/precedent_with_queries.jsonl"
output_file = "./dataset/sft/legal_ko/precedent_with_sum.jsonl"

print("데이터 로딩 중...")
lines = []
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()


total_lines = len(lines)
print(f"총 {total_lines}건 처리 시작...")

results = {}
error_count = 0
max_workers = 20

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(process_single_item, idx, line, law_llm): idx
        for idx, line in enumerate(lines)
    }

    # 진행 상황 표시
    for future in tqdm(as_completed(futures), total=total_lines, desc="Processing"):
        result = future.result()
        results[result["idx"]] = result
        if result["error"]:
            error_count += 1

print("\n결과 저장 중...")
with open(output_file, "w", encoding="utf-8") as f_out:
    for idx in sorted(results.keys()):
        result = results[idx]
        if result["error"]:
            f_out.write(result["data"])
        else:
            f_out.write(json.dumps(result["data"], ensure_ascii=False) + "\n")

print(f"\n완료!")
print(f"총 데이터: {total_lines}건")
print(f"성공: {total_lines - error_count}건")
print(f"오류 발생: {error_count}건")
print(f"저장 경로: {output_file}")
