import os
import dspy
import pandas as pd
from dspy import Example
import math
import json
import argparse
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2
from dspy.evaluate import Evaluate

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = ""

# DSPy 설정
dspy.configure(lm=dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000, cache=False))

LABELS = [
    "일치/불일치: 지문과 선지 내용이 일치하는지를 판단 및 글의 내용을 종합적으로 이해하고 글 전체를 포괄하는 핵심 내용을 파악하는 유형",
    "지문 기반 추론: 지문의 논리 흐름에 따라 추론할 수 있는 내용을 고르는 것으로, 논리적 비약 없이 지문의 내용을 바탕으로 타당하게 추론한 내용을 찾아야 하는 유형",
    "지문 기반 추론 (글쓴이의 견해): 글쓴이의 입장/주장/결론 또는 지문의 마지막 단락에서 말하고자 하는 주요 개념을 파악하여 풀이해야 하는 유형",
    "주장/요소 비교(ㄱ,ㄴ): ㄱ와 ㄴ에 대응되는 요소를 지문에서 연결하여 대립형으로 비교하여 풀이해야 하는 유형",
    "주장/요소 비교 (밑줄): 단락 안에서 해당하는 어구의 의미를 파악하여 풀이해야 하는 유형",
    "<보기> 기반 추론: <보기>가 제시되고, <보기>와 지문 내 논의의 문단과 맥락상 대입하여 풀이해야 하는 유형",
    "<보기> 기반 추론-ㄱ,ㄴ,ㄷ: 지문 외의 ㄱ,ㄴ,ㄷ <보기>를 제시 후 보기 문장의 유형 파악 후 지문 내 논의의 문단과 맥락상 대입하여 풀이해야 하는 유형",
]

TYPE_TO_LABEL = {
    "내용 이해 문제": "일치/불일치",
    "사실 확인 문제": "일치/불일치",
    "요소 이해 문제": "일치/불일치",
    "지문 기반 추론": "지문 기반 추론",
    "지문 기반 추론 (글쓴이의 견해)": "지문 기반 추론 (글쓴이의 견해)",
    "주장/요소 비교 (ㄱ, ㄴ)": "주장/요소 비교(ㄱ,ㄴ)",
    "주장/요소 비교 (밑줄)": "주장/요소 비교 (밑줄)",
    "<보기>기반 추론": "<보기> 기반 추론",
    "<보기>기반 추론-ㄱ,ㄴ,ㄷ": "<보기> 기반 추론-ㄱ,ㄴ,ㄷ",
}

def build_label_defs_text(labels):
    """라벨 정의 텍스트 생성"""
    return "\n".join(labels)

label_defs_text = build_label_defs_text(LABELS)

class ClassifyProblem(dspy.Signature):
    """주어진 문제 (질문/지문/선지)를 읽고 문제 유형 7개 중 하나의 유형으로 분류하라.
    반드시 아래 JSON 포맷으로 출력하라:
    {"sub_type": "<라벨 정확히 1개>", "confidence": 0~1, "rationale": "<한두문장 근거>"}"""
    problem = dspy.InputField(desc="문제 본문, 필요 시 지문/선택지 포함")
    label_defs = dspy.InputField(desc="각 라벨의 짧은 정의/키워드")
    answer = dspy.OutputField(desc="JSON만 출력 (설명 금지)")

df_test = pd.read_csv("./dataset/LEET/언어이해_final.csv")
# 2017-2023년 (train dataset)
df = df_test[:220]
df = df[df['유형'] != '이미지 활용 문제']
# 2024년 (val dataset)  
df_2 = df_test.iloc[220:250]
df_2 = df_2[df_2['유형'] != '이미지 활용 문제']
# 2025년 (test dataset)
df_3 = df_test.iloc[250:]
df_3 = df_3[df_3['유형'] != '이미지 활용 문제']

def compose_problem(row):
    # NaN 방지
    def safe(x): 
        return "" if (pd.isna(x) or (isinstance(x, float) and math.isnan(x))) else str(x)
    parts = []
    q = safe(row.get("질문"))
    p = safe(row.get("지문"))
    v = safe(row.get("보기"))
    c = safe(row.get("선택지"))
    if q: parts.append("[질문]\n" + q)
    if p: parts.append("[지문]\n" + p)
    if v: parts.append("[보기]\n" + v)
    if c: parts.append("[선택지]\n" + c)
    return "\n\n".join(parts)

def build_examples(dataframe):
    """DataFrame에서 Example 리스트 생성"""
    examples = []
    for _, row in dataframe.iterrows():
        raw_type = str(row["유형"]).strip()
        if raw_type not in TYPE_TO_LABEL:
            continue
        gold_label = TYPE_TO_LABEL[raw_type]
        problem_text = compose_problem(row)
        ex = Example(
            problem=problem_text,
            label_defs=label_defs_text,
            answer=json.dumps({
                "sub_type": gold_label,
                "confidence": 1.0,
                "rationale": ""
            }, ensure_ascii=False)
        ).with_inputs("problem", "label_defs")
        examples.append(ex)
    return examples

train_examples = build_examples(df)      # 2017-2023년
dev_examples = build_examples(df_2)      # 2024년
test_examples = build_examples(df_3)     # 2025년

print(f"테스트 데이터 개수: {len(test_examples)}")


class Router(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(ClassifyProblem)  # 또는 dspy.ChainOfThought(ClassifyProblem)

    def forward(self, problem, label_defs):
        out = self.classify(problem=problem, label_defs=label_defs)
        return out

def metric_fn(gold, pred, trace=None):
    def extract_subtype(result_str):
        try:
            return json.loads(result_str).get("sub_type", "").strip()
        except Exception:
            return ""
    
    # 안전하게 answer 추출
    try:
        if isinstance(gold, str):  # 문자열이면 바로 추출 
            gold_label = extract_subtype(gold) 
        elif hasattr(gold, 'answer'):      # 속성 존재 여부 파악 
            gold_label = extract_subtype(gold.answer)
        else:   
            gold_label = extract_subtype(str(gold)) # 문자열로 변환 
        
        if isinstance(pred, str):
            pred_label = extract_subtype(pred)
        elif hasattr(pred, 'answer'):
            pred_label = extract_subtype(pred.answer)
        else:
            pred_label = extract_subtype(str(pred))
        
        return int(gold_label == pred_label)
    except Exception as e:
        print(f"Metric error: {e}")
        return 0

program = Router()

tp = BootstrapFewShotWithRandomSearch(
    metric=metric_fn,
    max_bootstrapped_demos=3,   # 자동 생성할 few-shot 예시 개수 상한
    max_labeled_demos=5,
    num_candidate_programs=20,           # 후보 응답 수 (self-consistency 느낌)
    max_rounds=2,              # 랜덤 탐색 반복 횟수
)

compiled_program = tp.compile(
    program,
    trainset=train_examples,
    valset=dev_examples
)

evaluator = Evaluate(
    devset=test_examples,
    metric=metric_fn,
    num_threads=2,
    display_progress=True,
    display_table=10
)

result = evaluator(compiled_program)

# 점수 출력
if hasattr(result, 'metric'):
    print(f"\n테스트 정확도: {result.metric:.2%}")
else:
    print(f"\n평가 결과: {result}")
