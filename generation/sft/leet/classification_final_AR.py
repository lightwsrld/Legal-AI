import os, dspy
import pandas as pd
import json
import math
from dspy.evaluate import Evaluate
from dspy import Example
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

os.environ['OPENAI_API_KEY'] = ""

dspy.configure(lm=dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000, cache=False))

MAIN_CATEGORIES = [
    "언어추리: 언어로 제시된 정보나 원리로부터 언어적 추리를 통해 새로운 정보를 이끌어 낼 수 있는 능력 측정",
    "모형추리: 제시된 정보나 제약조건으로부터 기호, 그림, 표, 그래프와 같은 비언어적 모형을 사용하여 새로운 정보를 이끌어 낼 수 있는지를 측정",
    "논증: 논증분석, 논쟁 및 반론, 논증평가 및 문제해결 능력을 측정"
]

CATEGORY_SUBTYPES = {
    "언어추리": [
        "법규범: 주어진 규정이나 법조문의 내용을 올바르게 해석한 후 다양한 사례와 상황에 적용하고, 비교할 수 있는지를 묻는 유형",
        "인문: ",
        "사회: ",
        "과학기술: ",
    ],
    "모형추리": [
       "형식추리/수리추리: 논리학을 바탕으로 추론하는 유형 또는 수리 연산, 그래프 등을 통해 수리적인 비율의 증감이나 절대/상대치를 추론하는 유형",
       "논리게임 (배치 및 정렬): 제약조건 속에서 항목 배열하기, 항목 연결하기, 묶기 등 규칙, 조건들을 종합적-논리적으로 짜맞추어 부합하는 상황 및 결과를 도출해내는 유형"
       "논리게임 (참거짓): 제시문에 주어진 네 개의 진술 중 모순된 진술을 찾아 일부 정보를 확정하고 확정된 정보에 선택지에 주어진 정보를 추가하여 정오를 판단하는 유형"
    ],
    "논증": [
        "논증분석: 논증의 전제와 결론을 바탕으로 전제가 어떻게 결론을 뒷받침하는지 등의 논리적 관계를 파악해 전체 글이 어떤 식으로 전개되는지 구조도를 맞추는 유형",
        "논쟁/반론: 둘 또는 다수의 대화가 제시되고, 각 입장에 대한 기본적인 이해나 상대방에 대한 반론의 근거나 논리적 수단을 물어보는 유형 또는 주어진 주장이나 논증에 대해 반박하는 능력을 측정하는 유형",
        "논증평가/문제해결: 하나의 논증에 대하여 종합적으로 평가하고 그러한 평가의 원리 내지 가정을 파악하는 유형 또는 하나 혹은 둘 이상의 주장이나 가설을 제시하고 새로운 경험적 증거나 새로운 정보에 의해 지문의 논지를 강화하거나 약화하는 선지를 찾는 유형 또는 지문에 주어진 가설이나 정보가 모순되거나 주어진 상황에 부합하지 않을 때, 이러한 가설이나 정보를 수정하지 않고도 설명할 수 있는 정보를 찾는 유형",
    ]
}

SUBTYPE_TO_CATEGORY = {
    "법규범": "언어추리",
    "인문": "언어추리",
    "사회": "언어추리",
    "과학기술": "언어추리",
    "형식추리/수리추리": "모형추리",
    "논리게임 (배열)": "모형추리",
    "논리게임 (참거짓)": "모형추리",
    "논증분석": "논증",
    "논쟁/반론": "논증",
    "논증평가/문제해결": "논증",
}

TYPE_TO_LABEL = {
    "법규범": "법규범",
    "인문": "인문",
    "사회": "사회",
    "과학기술": "과학기술",
    "형식추리": "형식추리/수리추리",
    "수리추리": "형식추리/수리추리",
    "논리게임(배열))": "논리게임 (배열)",
    "논리게임(참거짓)": "논리게임 (참거짓)",
    "논증분석": "논증분석",
    "논쟁/반론": "논쟁/반론",
    "논증평가": "논증평가/문제해결",
}

class ClassifyMainCategory(dspy.Signature):
    """주어진 문제를 읽고 대분류(언어추리/모형추리/논증) 중 하나로 분류하라.
    반드시 아래 JSON 포맷으로 출력하라:
    {"main_category": "<언어추리 또는 모형추리 또는 논증>", "confidence": 0~1, "rationale": "<한두문장 근거>"}"""
    problem = dspy.InputField(desc="문제 본문")
    category_defs = dspy.InputField(desc="각 대분류의 정의")
    answer = dspy.OutputField(desc="JSON만 출력")

class ClassifySubType(dspy.Signature):
    """주어진 문제와 대분류를 바탕으로 세부 유형(sub_type)을 분류하라.
    반드시 아래 JSON 포맷으로 출력하라:
    {"sub_type": "<세부 유형>", "confidence": 0~1, "rationale": "<한두문장 근거>"}"""
    problem = dspy.InputField(desc="문제 본문")
    main_category = dspy.InputField(desc="1단계에서 분류된 대분류")
    subtype_defs = dspy.InputField(desc="해당 대분류의 세부 유형 정의")
    answer = dspy.OutputField(desc="JSON만 출력")

df_test = pd.read_csv("./dataset/LEET/추리논증_final.csv")
# 2017-2023년 
df = df_test[:270]
df_2 = df_test.iloc[270:310]
df_3 = df_test.iloc[310:]


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
        sub_type = TYPE_TO_LABEL[raw_type]
        if sub_type not in SUBTYPE_TO_CATEGORY:
            continue
        main_category = SUBTYPE_TO_CATEGORY[sub_type]
        combined_label = f"{main_category}-{sub_type}"
        problem_text = compose_problem(row)
        ex = Example(
            problem=problem_text,
            answer=json.dumps({
                "sub_type": combined_label,
                "confidence": 1.0,
                "rationale": ""
            }, ensure_ascii=False)
        ).with_inputs("problem")
        examples.append(ex)
    return examples

train_examples = build_examples(df)      # 2017-2023년
dev_examples = build_examples(df_2)      # 2024년
test_examples = build_examples(df_3)     # 2025년

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
    

class TwoStageClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify_main = dspy.Predict(ClassifyMainCategory)
        self.classify_sub = dspy.Predict(ClassifySubType)
    
    def forward(self, problem):
        # 1단계: 대분류
        category_defs_text = "\n".join([f"- {cat}" for cat in MAIN_CATEGORIES])
        main_result = self.classify_main(
            problem=problem,
            category_defs=category_defs_text
        )
        
        try:
            main_json = json.loads(main_result.answer)
            main_category = main_json["main_category"].strip()
        except:
            main_category = "언어추리"
        
        # 2단계: 세부 유형
        if main_category in CATEGORY_SUBTYPES:
            subtype_defs_text = "\n".join([f"- {s}" for s in CATEGORY_SUBTYPES[main_category]])
        else:
            subtype_defs_text = "정의 없음"
        
        sub_result = self.classify_sub(
            problem=problem,
            main_category=main_category,
            subtype_defs=subtype_defs_text
        )
        
        # 최종 결과를 "main_category-sub_type" 형식으로 반환
        try:
            sub_json = json.loads(sub_result.answer)
            sub_type = sub_json["sub_type"].strip()
            
            # 이미 "main_category-sub_type" 형식이 아니면 조합
            if "-" not in sub_type:
                combined = f"{main_category}-{sub_type}"
            else:
                combined = sub_type
            
            return dspy.Prediction(
                answer=json.dumps({
                    "sub_type": combined,
                    "confidence": sub_json.get("confidence", 1.0),
                    "rationale": sub_json.get("rationale", "")
                }, ensure_ascii=False)
            )
        except:
            return dspy.Prediction(answer=sub_result.answer)
        

teleprompter = BootstrapFewShotWithRandomSearch(
    metric=metric_fn,
    max_bootstrapped_demos=5,  # few-shot 예제 개수
    max_labeled_demos=10,       # trainset에서 몇 개 예시로 부트스트랩할 지 
    num_candidate_programs=20, # 샘플링할 후보 프로그램 수
    num_threads=2              # 병렬 처리
)

# 컴파일 (최적화)
classifier = TwoStageClassifier()
optimized_classifier = teleprompter.compile(
    classifier, 
    trainset=train_examples,
    valset=dev_examples  
)

evaluator = Evaluate(
    devset=test_examples,
    metric=metric_fn,
    num_threads=2,
    display_progress=True,
    display_table=40
)

result = evaluator(optimized_classifier)

# 점수 출력
if hasattr(result, 'metric'):
    print(f"\n테스트 정확도: {result.metric:.2%}")
else:
    print(f"\n평가 결과: {result}")
