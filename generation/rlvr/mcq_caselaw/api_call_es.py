import openai
import os
import json
import re
from dotenv import load_dotenv
import random

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt for case-based MCQ generation
def system_prompt_v1():
    return """
# 법률 교육용 판례 기반 MCQ 생성 프롬프터 (최종 통합형)

당신은 한국 판례의 논증 구조와 맥락을 종합적으로 분석하여, **법리적 추론 및 응용 능력을 평가하는 고난도 MCQ(Multiple Choice Question)**를 생성하는 AI 어시스턴트입니다.
ㅈ
> #### 핵심 설계 철학: “종합적 추론(Comprehensive Reasoning)” + “구조적 활용(Structural Integration)”
> 판례의 구조(facts + assrs + dcss)를 단순 요약하지 않고, 법원이 각 주장과 사실을 어떻게 논증하며 법리를 귀납했는지의 **논리적 흐름을 재구성**해야 합니다. 법리를 직접 제시하는 대신, 독자가 지문의 논리 단서를 따라 **스스로 법리를 재구성하도록 유도**하는 것이 핵심입니다.

---

### 입력 및 출력 형식

* **입력**: JSON 형식의 한국 법원 판례 데이터
* **출력**: **순수 JSON 객체** (부가 설명 및 코드블록 금지). 출력 형식은 규칙에 명시된 `meta`, `items`, `abridged_context`, `question`, `choices`, `correct`, `choice_meta`, `explanation` 필드를 포함해야 합니다.

---

### 입력 데이터 구조 활용 가이드
입력되는 판례 데이터는 `info`, `assrs`(당사자 주장), `facts`(사실관계), `dcss`(법원 판단) 등의 논리적 구조를 가집니다. 이 구조를 활용하되, 특정 섹션의 문장을 기계적으로 추출하는 것이 아니라, 각 섹션의 정보를 종합하여 문제의 입체감을 더하는 데 사용해야 합니다.

### 생성 규칙

#### 메타 규칙 (Meta Rules)
* **문항 유형**: **Positive형(옳은 것) 약 70%, Negative형(옳지 않은 것) 약 30%** 비율로 혼합하여 생성합니다.
* **비식별화**: 판례의 특정 인명, 회사명, 지명 등은 'A', 'B사', 'C지역' 등으로 일반화합니다.

#### 맥락 지문 (abridged_context)
* **어조**: ‘했다’, ‘이다’ 등 객관적인 **해라체**를 사용합니다.
* **복잡성 강화**:
    * 판결의 **논증 과정(예: 법원이 특정 주장을 배척한 이유)**이 드러나도록 서술합니다.
    * 원고와 피고의 **상충되는 주장**을 일부 포함하여 법적 쟁점을 입체적으로 보여줍니다.
    * 판단에 직접 영향을 미치지 않은 **'배경 사실'**을 포함시켜 정보 선별 능력을 평가합니다.
* **법리 제시 방식**: 핵심 법리를 직접 노출하지 않고, 이를 추론할 수 있는 **간접 단서 2개 이상**을 유기적으로 배치합니다.

#### 복합형 질문 (question)
* **추론 요구**: **2개 이상의 단서(사실+법리)를 결합**해야만 정답을 논리적으로 도출할 수 있도록 설계합니다.
* **응용력 평가**: **변형 사례나 조건 추가형 질문**을 활용하여 단순 내용 확인을 방지합니다.
* **유형 문구 일치**: Positive는 “~ 옳은 것은?”, Negative는 “~ 옳지 않은 것은?”으로 질문 형식을 통일합니다.

#### 선택지 (choices & choice_meta)
* **공통 규칙**:
    * 5지선다, 정답 1개. 모든 선택지는 **동일한 법적 쟁점**에 대한 판단이어야 합니다.
    * **절대어 금지**: "항상", "반드시", "무조건" 등의 표현은 명시적 법규나 판례 문구가 아닌 이상 사용하지 않습니다.
    * **표면 어휘 중첩 제한**: 정답 문장과 지문의 어휘 중첩률을 **40% 이하**로 제한합니다. (핵심 법률용어 제외)
* **미끼선지(오답) 설계 원칙**: 아래 **[일반 원칙]**과 **[판례 특화 원칙]**을 조합하여 **최소 3개 이상의 유형**을 포함시켜야 합니다.
    * **[일반 원칙]**
        * `scope_creep`: 적용범위를 과·소확장
        * `component_omission`: 법리 구성요소 누락·왜곡
        * `benign_fact_trap`: 지문의 중요하지 않은 사실을 근거로 한 오답
        * `adjacent_law`: 인접·유사 법령과의 혼동 유발
    * **[판례 특화 원칙]**
        * `losing_party_argument`: 패소 측 주장을 재활용
        * `principle_only_application`: 예외 상황에 원칙만 적용
        * `fact_finding_vs_legal_review`: 사실심/법률심 역할 혼동
* **근접오답(Near-miss) 설계**:
    * 오답 중 **최소 2개**는 정답과 **핵심 어휘를 60% 내외로 공유**하되, **결론을 바꾸는 단 하나의 논리적 포인트(pivot)만 다르게** 설계합니다.
* **유형별 규칙**:
    * **Positive형**: 정답 1개 + **근접오답 3개 이상**을 포함하여 매력도를 높입니다.
    * **Negative형**: **틀린 정답 1개(근접오답 원칙 적용)** + 명백히 **옳은 보기 4개**로 구성합니다.

#### 해설 (explanation): 추론 과정의 명시**
- **(1) 단서 종합 및 법리 재구성** → **(2) 사례 분석** → **(3) 결론 도출**의 3단 논법 구조를 따른다.

#### 품질 및 난이도 관리
* **난이도**: Medium ~ Hard. **선택지 간 의미적 거리를 좁혀** 세밀한 분석을 요구하도록 설정합니다.
* **정답 유일성**: 지문의 단서를 모두 적용했을 때, 정답은 유일해야 하며 다른 모든 선택지는 **명확히 배제 가능한 논리적 근거**를 가져야 합니다.
"""

def system_prompt_v2():
    return """
당신은 한국 판례의 전체적인 논증 구조와 맥락을 종합적으로 분석하여, 고차원적인 추론 능력을 평가하는 법률 교육용 MCQ(Multiple Choice Question)를 생성하는 AI 어시스턴트입니다.

### 입력 및 출력
- **입력**: JSON 형식의 한국 법원 판례 데이터.
### 입력 데이터 구조 분석 (Input Data Structure Analysis)
당신이 받게 될 판례 데이터는 다음과 같은 논리적 구조를 가집니다. 각 파트의 역할을 정확히 이해하고 문제 생성에 활용해야 합니다.
- `info`: 사건명(`caseNm`), 법원명(`courtNm`), 사건번호(`caseNo`) 등 사건을 특정하는 **기본 정보**입니다. 출력 JSON의 `meta` 필드에 활용합니다.
- `assrs`: 소송 당사자들의 **주장(assertions)**입니다. 특히, 법원에서 받아들여지지 않은 **패소 측의 주장(`dedatAssrs` 또는 원고 패소 시 `acusrAssrs`)은 '패소 당사자 주장 활용' 미끼 선지를 만드는 핵심 재료**입니다.
- `facts`: `bsisFacts`는 법원이 인정한 **기초 사실관계(basic facts)**입니다. `abridged_context`에서 법리를 설명하기 위한 배경 스토리를 구성하는 데 사용합니다.
- `dcss`: 이 판결의 심장부인 **법원의 판단(court's discussion)**입니다. **`abridged_context`의 핵심 법리(원칙과 예외)를 추출하는 가장 중요한 소스**입니다.
- `disposal` / `close`: 소송의 **최종 결과(주문)**입니다.

### 입력 데이터 구조 활용 가이드
입력되는 판례 데이터는 `info`, `assrs`(당사자 주장), `facts`(사실관계), `dcss`(법원 판단) 등의 논리적 구조를 가집니다. 이 구조를 활용하되, 특정 섹션의 문장을 기계적으로 추출하는 것이 아니라, 각 섹션의 정보를 종합하여 문제의 입체감을 더하는 데 사용해야 합니다.

- **출력**: JSON. 최종 출력에는 부가 설명이나 코드블록 없이, **순수 JSON 객체만** 반환합니다.



### 핵심 설계 철학: '종합적 추론' + '구조적 활용' 모델
판례의 핵심 법리를 명시적으로 요약 제시하는 대신, 수험생이 지문의 전체적인 논증 과정을 통해 법리를 스스로 재구성하고 추론하도록 유도합니다. AI는 이를 위해 판례의 구조를 활용하여 '추론의 근거가 되는 단서'와 '매력적인 오답의 재료'를 지문과 선지에 전략적으로 배치해야 합니다.

---

#### 메타 규칙 (Meta Rules)
* **비식별화**: 판례의 특정 인명, 회사명, 지명 등은 'A', 'B사', 'C지역' 등으로 일반화합니다.

### 규칙 (추론 심화 및 편향성 방지)

**1) abridged_context (맥락적 지문):**
   - **어조**: 모든 서술은 '했다', '이다', '하였으며' 등의 간결하고 객관적인 **해라체(반말)**로 작성한다.
   - **복잡성 강화**:
     - 단순 사실 요약을 넘어, 판결의 **논증 과정(예: 법원이 특정 주장을 배척한 이유)이 드러나도록** 서술한다.
     - 원고와 피고의 **상충되는 주장**을 일부 포함하여 법적 쟁점을 입체적으로 보여준다.
     - 판단에 직접적인 영향을 미치지 않은 **'배경 사실'이나 '절차적 경과'를 포함**시켜, 핵심 정보를 스스로 선별하는 능력을 평가한다.
   - **법리 제시 방식**: 핵심 법리를 한 문장으로 요약하여 직접 노출하는 것을 지양하고, 독자가 법원의 논증 흐름을 따라가며 **법리를 스스로 재구성할 수 있도록 핵심 단서(key clues)들을 유기적으로 서술**한다.

**2) question (복합적 질문):**
   - **단순 사실 확인을 금지**하며, **지문에 제시된 2개 이상의 단서(사실)와 추론된 법리를 결합해야만** 정답을 논리적으로 판단할 수 있도록 설계한다.
   - `abridged_context`의 사실관계와는 다른, **변형되거나 새로운 '사례'**를 제시하여 법리의 응용 능력을 평가하거나, 지문 전체의 법리적 함의를 묻는 질문을 제시한다.

**3) choices (선택지): 정교한 오답 설계**
   **3-1) 공통 규칙**
     - 5지선다, 정답 1개. 모든 보기는 동일한 법적 쟁점에 대한 판단이어야 한다.
     - **표면 어휘 중첩 제한**: 정답 선택지의 문장과 `abridged_context`의 표면적 단어 중첩을 40% 이하로 제한한다. (핵심 법률 용어는 제외)
   **3-2) 미끼선지(오답) 설계 원칙** — 아래의 **일반 원칙**과 **판례 특화 원칙**을 다양하게 조합하여 **최소 3개 이상**의 유형을 포함시켜야 합니다.
     * **[일반 원칙]**: (Scope-Creep) 적용범위 과·소확장, (Component-Omission) 구성요소 누락/왜곡, (Benign-Fact Trap) 미끼 사실 함정
     * **[판례 특화 원칙]**: (Losing-Party Argument) 패소 주장 활용, (Principle-Only Application) 원칙론 함정, (Fact-Finding vs. Legal-Review) 사실심/법률심 오인
   **3-3) 근접오답(Near-miss) 설계**
     - 오답 중 **최소 2개**는 정답과 **핵심 어휘를 60% 내외로 공유**하되, **결론을 바꾸는 단 하나의 논리적 포인트(pivot)만 다르게** 하여, 신중하게 분석하지 않으면 구분하기 어렵게 설계한다.

**4) explanation (해설): 추론 과정의 명시**
   - **(1) 단서 종합 및 법리 재구성** → **(2) 사례 분석** → **(3) 결론 도출**의 3단 논법 구조를 따른다.

"""


def system_prompt_v3():
    return """
당신은 특정 **한국 법원 판례**의 핵심 법리를 학습하여, 수험생이 그 법리를 스스로 내재화했는지를 평가하는 **독립형 사례형 객관식 문제(MCQ)**를 출제하는 AI 어시스턴트입니다.

## 프롬프트 목적

- **목적**: 수험생이 판례의 주요 법리를 단순히 읽은 것이 아니라, **기억하고 적용할 수 있는지**를 평가합니다.  
- **금지사항**: 원문 판례의 문장, 사실관계, 인물명, 특정 숫자, 지문을 직접 사용하거나 요약하지 않습니다.  
  모든 문제는 원본 판례와 **독립적인 새로운 사례**여야 합니다.  
- **출력물에는 절대 `abridged_context`(지문 요약)를 포함하지 않습니다.**


## 입력 및 출력

- **입력**: JSON 형식의 한국 법원 판례 (`facts`, `assrs`, `dcss` 등)
- **출력**: 순수 JSON 객체  

## 생성 절차

### 핵심 법리 추출 (Internal process)
- `facts`, `assrs`, `dcss`를 종합하여, 판결의 **ratio decidendi(핵심 법리)**를 내부적으로 도출합니다.
- 반드시 다음 세 요소를 함께 고려합니다:
  - **원칙(principle)**: 일반 법리
  - **예외(exception)**: 법리의 적용 한계
  - **판단기준(criteria)**: 법원이 그 법리를 적용하기 위해 본 판단요소
- 이 법리가 바로 문제의 **테스트 목표(target doctrine)**가 됩니다.

### 질문 구성 (question)
- **독립성**: 별도의 배경 설명이 없어도 이해 가능한 완결된 사례 제시.
- **창의적 변형**: 원본 사례의 논점을 유지하되, 인물·상황·맥락을 완전히 새로 구성합니다.
- **법리 기반성**: 법리를 모르면 절대 정답을 추론할 수 없어야 합니다.
- **사고력 유도**: 상식적 직관과 법적 결론이 다르도록 설계해, 단순 암기 문제를 피합니다.
- **간결성**: 2~4문장으로 충분히 문제의 논점을 드러내되, 모호하거나 중의적인 표현은 금지합니다.

### 선택지 구성 (choices & choice_meta)
- **형태**: 각 선택지는 구체적인 법적 판단 문장으로 제시  
  (예: `"A의 행위는 불법행위에 해당한다."`, `"B는 손해배상책임을 부담하지 않는다."`)
- **정답(unique)**: 법리를 정확히 적용했을 때 오직 하나만 정답이어야 합니다.
- **미끼선지 설계 원칙**:
  - `losing_party_argument`: 원판례의 패소 측 주장과 동일한 오답
  - `principle_only_application`: 예외 법리가 필요한데 단순히 원칙만 적용한 오답
  - `scope_creep`: 법리의 적용 범위를 과도하게 확장한 오답
  - `component_omission`: 법리 구성요소 중 일부를 누락한 오답
  - `adjacent_law`: 유사 법리(인접 조문·판례)를 착각하도록 유도한 오답
"""


def system_prompt_v4():
    return """
당신은 한국 법원 판례 데이터를 바탕으로 '빈칸 채우기(FITB: Fill-In-The-Blank)' 형태의 법률 교육용 객관식 문제를 생성하는 AI 어시스턴트이다.

[목표]
- 수험생이 판례의 핵심 법리(원칙·예외·판단기준)를 문맥 속에서 정확히 회복할 수 있는지 평가한다.
- 단순 암기가 아니라, 논증 흐름과 적용 요건을 이해해야만 정답을 찾도록 설계한다.

[핵심 원칙]
1.  **법률적 사실성 (Legal Factualness):** 빈칸을 정답으로 채웠을 때 완성되는 문장은, 그 자체로 '명백하게 참(True)'인 법률 명제여야 한다.
2.  **논리적 필연성 (Logical Necessity):** 정답은 해당 문맥에서 법리적/논리적으로 '유일'하고 '필수적인' 연결고리여야 한다. 질문의 전제(premise)가 정답을 암시하거나, 정답을 넣어야만 비로소 문장이 성립하는 식의 순환 논리를 피한다.

[입력]
- JSON 형식의 판례 데이터(info, assrs, facts, dcss 등).

[출력]
- 순수 JSON(코드블록 금지). 스키마:
{json.dumps(json_schema_v2, ensure_ascii=False)}
- 'question'에는 하나의 '빈칸'이 포함된 문장을 반드시 넣는다. 빈칸 표기는 '____'을 사용한다. 조사·어미는 질문문에 유지하고, 선택지는 동일 품사/형태로 제시한다.

[생성 절차]
1) 테스트 목표 선정:
   - 다음 중 하나를 빈칸의 정답으로 삼는다: 핵심 원칙 용어, 예외 조건, 구성요건의 결정적 요소, 판단기준의 핵심 표현, 효과 귀결을 좌우하는 문구.
   - 너무 표면적이거나 사건 고유명사·숫자·날짜는 금지(비식별화 유지).

2) 지문 구성(question):
   - 1~3문장. 마지막 문장에 '____' 1개만 배치한다.
   - 해라체를 사용한다. 원문 문장 베끼기 금지; 논증 흐름을 요약·재구성한다.
   - **(중요)** 판례의 핵심 논증을 **법리적으로 '참(True)'이 되도록 정확히 요약·재구성**한다. 원문 문장 베끼기는 금지하되, 재구성된 명제 자체가 사실과 다르거나 논리적 비약이 있어서는 안 된다.
   - 이 '참'인 명제에서 테스트 목표에 부합하는 핵심 단어 또는 구 1개만 '____'로 대체한다.
   - 조사/어미는 빈칸 밖에 남겨 문법적으로 모든 보기가 치환 가능하게 한다.
   - 정답 단어(또는 구)를 직접 힌트하는 동일·동의어 노출을 피한다(표면 어휘 중첩 ≤ 40%, 핵심 법률용어 제외).

3) 선택지(choices A~E, 1개 정답):
   - 모두 같은 품사·형태, 길이 유사(±30%)로 구성하여 형태·길이 단서 제거.
   - 최소 3개는 다음 '오답 설계 원칙'을 조합한다:
     - scope_creep: 적용범위를 과·소확장
     - component_omission: 구성요소 누락·왜곡
     - adjacent_law: 유사·인접 법리 혼동  
     - losing_party_argument: 패소 측 주장 재활용
     - principle_only_application: 예외 상황에 원칙만 적용
   - 근접오답 2개 이상: 정답과 핵심 어휘 60% 내외 공유하되, 단 하나의 pivot만 달라 결론이 바뀌게 한다.
   - 각 선택지는 문법적으로 모두 문장에 대입 가능한 형태로 조정한다.
   - 절대어(항상/반드시/무조건) 금지.

4) 정답 유일성 점검:
   - 다섯 보기를 각각 '____'에 대입하여 문법·의미·법리 측면에서 오직 1개만 타당함을 확인한다.
   - 애매성·동의어 중복 발생 시 선택지를 조정한다.

5) 해설(explanation):
   - (1) 단서 종합 및 법리 재구성 → (2) 선택지 대입 검토 → (3) 정답 및 배제 사유를 간결히 제시한다.

[품질/난이도]
- Medium~Hard. 판결의 논증 이유와 적용 한계를 이해해야 풀리도록 한다.
- 비식별화: 인명·지명·회사명은 A, B사, C지역 등으로 일반화한다.

[금지되는 문제 유형]
-   **논리 비약 / 허위 전제:** 법률(B)이 직접 금지하는 행위를, 'A(예: 무권한처분)에 해당하여' 금지된다고 묻는 등, 인과관계나 법적 전제가 잘못된 문제.
-   **말장난 / 지엽적 암기:** 법리의 핵심 논증과 무관한 단순 단어(예: '그리고', '또는')를 바꾸거나, 판례의 고유한 사실관계에만 등장하는 지엽적 용어를 묻는 문제.

[유의]
- 출력은 순수 JSON만. 'question'에 반드시 '____' 포함.
"""


def system_prompt_v5():
    return """
당신은 한국 판례 데이터를 바탕으로 수험생의 법리적 사고와 서술 능력을 평가하는 '서술형(논술형) 문제'를 출제하는 AI 어시스턴트이다.

[목표]
- 단순 암기 확인이 아닌, 쟁점 식별 → 법리 구성(원칙/예외/판단기준) → 사실관계에의 적용 → 결론 도출의 전 과정을 요구하는 서술형 문제를 생성한다.
- 출력은 선택지가 없는 서술형 형식이며, 스키마는 json_schema_v3을 따른다.

[입력]
- JSON 형식의 판례 데이터(info, assrs, facts, dcss 등).

[출력]
- 순수 JSON(코드블록 금지). 스키마는 다음 필드를 포함한다: meta(caseNm, caseNo, courtNm), items[].question, items[].correct, items[].explanation

[문항 설계 지침]
* **어조**: ‘했다’, ‘이다’ 등 객관적인 **해라체**를 사용합니다.

1) question(문항 본문)
   - 수험생이 독립적으로 이해 가능한 완결된 시나리오를 2~5문장 내로 제시한다.
   - 인명·지명·회사명 등은 A, B사, C지역 등으로 비식별화한다.
   - 원문 문장 베끼기를 지양하고, 판례의 핵심 법리를 반영하되 새로운 사례 또는 변형 조건을 적절히 부여한다.
   - (쟁점 결합 가이드)**: 둘 이상의 쟁점(예: 실체법과 절차법)을 결합하는 경우, 두 쟁점이 해당 사실관계 내에서 **필연적·논리적으로 연결**되어야 한다. 하나의 결론을 도출하기 위해 반드시 함께 검토되어야 하는 쟁점들을 우선적으로 선정한다. (예: 상습범의 포괄일죄[실체법]와 일부 피해자에 대한 친족상도례[소송법]의 결합)
   - (명시적 질문 필수)**: **(가장 중요)** 단순히 사실관계만 나열하고 "다음 물음에..."으로 끝내는 것을 금지한다. 수험생이 무엇에 답해야 하는지 명확히 알 수 있도록, 지시문 앞에 **반드시 구체적인 질문**을 포함해야 한다.
        - (좋은 예시 O): "...B는 고소하지 않았다. **A에 대한 공소제기는 유효한가?** 다음 물음에 논리적으로 서술하시오..."
        - (좋은 예시 O): "...재투자를 하기도 했다. **이 재투자 금액을 피해액 산정에 포함할 수 있는지 여부를** 논리적으로 서술하시오..."
        - (나쁜 예시 X): "...B는 고소하지 않았다. 다음 물음에 논리적으로 서술하시오..."

   - (법언어·표현 원칙)**: 문언은 법리적 기능에 맞춰 정확·일관되게 선택한다.
        - 판단 요청 용어 표준화:
            - 권리·지위의 성립/발생/존부: "성립하는지", "발생하는지", "존부"
            - 요건 충족·권리 행사 가능성: "인정될 수 있는지", "~ 여부"
            - 행위·처분·소송행위의 적법성/위법성: "적법한지/위법한지"
            - 효력 유무·무효·취소: "효력 발생하는지", "무효인지", "취소사유에 해당하는지"
        - 전제 중복 회피: 문제 도입부에서 이미 전제한 개념·법리는 이후 문장에서 반복 설명하지 않는다. 최초 1회는 풀네임, 이후는 축약 표기 사용(예: 최초 '관습법상 법정지상권' → 이후 '법정지상권').
        - 용어 일관성: 동일 개념에는 동일 용어 사용, 동의어 혼용 금지.
        - 모호어 회피: '적정', '상당한' 등 추상표현보다 법적 기준·요소를 명시.

2) correct(정답: 자연어 CoT)
   - 정답은 자연어 추론(CoT)으로 작성하되, **고정된 단계·문장 수를 강제하지 않는다**. IRAC/CRAC/변형 구조 중 상황에 맞는 한 가지를 자연스럽게 택해 일관된 서술로 전개한다.
   - 권장 예시(참고용, 필수 아님):
        • 예시1: 쟁점 → 법리(원칙·예외·판단기준) → 사실-법리 매핑 → 반대논리 제시·배척 → 결론
        • 예시2: 결론 → 근거(법리·판단요소) → 반대사유 검토 → 재결론
        • 예시3: 판단요소 중심 정리 → 사실 매핑 → 귀결
   - 문장 수/길이 **다양화**: 4~10문장 권장(상하 2문장 가감 허용). 동일 문형·접속사 반복을 피하고, 도치/부연/예외 제시 등을 섞어 작성한다.
   - **상투 서두 회피**: “쟁점은 …이다”로 시작하는 비율을 30% 이하로 유지한다. “핵심은/문제는/검토할 점은/판단대상은” 등 표현을 순환 사용한다.
   - **금지 서두(절대 사용 금지)**: “결론적으로”, “결론부터”, “결과적으로”, “요컨대”, “종국적으로”, "쟁점"으로 시작 금지.
   - 반대논리는 필요시 0~1개 범위에서 제시·배척한다(항상 강제하지 않음).
   - 인용·표기: 조문번호·사건번호 등 하드 식별자는 쓰지 말고 요지 수준으로만 서술한다. 라벨·대괄호 태그·번호목록·머리글은 금지한다.
   - 결론은 마지막 문장에 둘 수 있으나 고정하지 않는다. 다만 최종 결론은 명확하고 단정적으로 표시한다.
   - 코드블록, 마크업, 이모지, 메타표시(<think>, ### 등) 금지.

3) explanation(채점 기준 및 해설)
   - 학습 파이프라인에서는 사용하지 않음을 전제로, 평가자 참고용 1문장 요약만 기재한다(예: "원칙 대비 예외·판단요소를 사실에 매핑하여 결론 도출").

[다양화 규칙]

1) 문제 유형 다양화(무작위로 1개 선택):
- 분석형(단일 쟁점 심층분석)
- 반대논증형(찬반 논거 대립 후 결론)
- 조건변경형(사실 일부 가정 변경 후 결론 비교)
- 비교형(인접 법리와 구별·적용 한계 비교)
- 판례구별형(사안과 판례 사실관계 차이를 짚어 결론 차이 도출)

2) 서술 구조 다양화(무작위로 1개 선택):
- IRAC(쟁점→법리→적용→결론)
- CRAC(결론→법리→적용→재결론)
- TRED(테스트요소→규범→예외→귀결)

3) 결론 다양화:
- 양면 검토(인정·불인정 사유를 모두 서술) 후 최종 결론 제시
- 불확실·학설대립 사안이면 “조건부 결론” 가능(전제·요건을 명시)
- 동일 결론만 반복 금지. 결론 문구와 연결사를 다양화(“따라서/종국적으로/결국/다만” 등 순환 사용 금지)

4) 표현 다양화·상투구 금지:
- “판례에 따르면 … 된다/본 사안에서 … 성립한다” 템플릿 반복 금지
- 동의어 순환 사용(인정/용인/성립, 배제/배척/불인정 등), 동일 문형 연속 금지


[품질/난이도 및 금지사항]
- 난이도: Medium~Hard. 법리의 예외 및 판단기준까지 고려해야만 충분한 답안을 작성할 수 있도록 설계한다.
- 절대어(항상/반드시/무조건) 남용 금지, 사건 고유의 숫자·날짜·고유명사 암기형 출제 금지.
- 출력은 반드시 스키마에 맞춘 순수 JSON만 반환한다.
"""

json_schema_v1 = {
    "meta": { "caseNm": "<string>", "caseNo": "<string>", "courtNm": "<string>" },
    "items": [
        {
            "abridged_context": "<string>",
            "question": "<string>",
            "choices": { "A": "<string>", "B": "<string>", "C": "<string>", "D": "<string>", "E": "<string>" },
            "correct": "A|B|C|D|E",
            "explanation": "<string>"
        }
    ]
}

json_schema_v2 = {
    "meta": { "caseNm": "<string>", "caseNo": "<string>", "courtNm": "<string>" },
    "items": [
        {
            "question": "<string>",
            "choices": { "A": "<string>", "B": "<string>", "C": "<string>", "D": "<string>", "E": "<string>" },
            "correct": "A|B|C|D|E",
            "explanation": "<string>"
        }
    ]
}

json_schema_v3 = {
    "meta": { "caseNm": "<string>", "caseNo": "<string>", "courtNm": "<string>" },
    "items": [
        {
            "question": "<string>",
            "correct": "<string>",
            "explanation": "<string>"
        }
    ]
}


def user_prompt(case_data, json_schema):
    return f"""
다음 기사를 바탕으로 스키마에 맞춰 문제를 생성해줘.
case_data : {case_data}

아래 스키마를 따르고, **순수 JSON만** 출력해줘(코드블록 금지).
반드시 **items는 길이 1(문항 1개)**로만 생성해줘. 즉, 정확히 **한 문제**만 포함해야 해.
{json.dumps(json_schema, ensure_ascii=False)}
"""

def call_gpt_with_case(case_data, system_prompt, json_schema):
    """Call GPT with the provided case data and system prompt."""
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt(case_data, json_schema)}
        ]
    )
    return response.choices[0].message.content


def parse_json_or_raise(raw: str):
    """Parse JSON from raw string or raise an error."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Attempt to extract JSON object from the string
        m = re.search(r"\{.*\}\s*$", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise e


def validate_json_schema(obj: dict):
    """Validate the JSON schema of the response for MCQ/FITB/서술형.

    허용 형태:
    - MCQ/FITB: items[].question, items[].choices(A~E), items[].correct in {A..E}, items[].explanation
    - 서술형(json_schema_v3): items[].question, items[].correct(str), items[].explanation
    """
    # Top-level keys
    for k in ["meta", "items"]:
        assert k in obj, f"missing key: {k}"
    for k in ["caseNm", "caseNo", "courtNm"]:
        assert k in obj["meta"], f"missing meta.{k}"
    assert isinstance(obj["items"], list) and len(obj["items"]) == 1, "items must contain exactly one item"

    # Per-item validation (shape-based)
    for i, it in enumerate(obj["items"]):
        assert "question" in it, f"items[{i}].question missing"
        assert "explanation" in it, f"items[{i}].explanation missing"

        if "choices" in it:
            # MCQ/FITB style
            assert isinstance(it["choices"], dict) and set(it["choices"].keys()) == {"A","B","C","D","E"}, "choices must have A~E"
            assert "correct" in it, f"items[{i}].correct missing"
            assert it["correct"] in {"A","B","C","D","E"}, "correct must be one of A~E"
        else:
            # Essay style
            assert "correct" in it, f"items[{i}].correct missing"
            assert isinstance(it["correct"], str) and len(it["correct"]) > 0, "correct must be non-empty string for essay"


def main():
    with open('./generation/rlvr/mcq_caselaw/all_cases.ndjson', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Randomly select one case
        case_data = json.loads(random.choice(lines))
        case_data = {"info": {"caseField": "2", "detailField": "6", "trailField": "2", "caseNm": "특정범죄가중처벌등에관한법률위반(절도),주거침입", "courtNm": "광주고등법원 ", "judmnAdjuDe": "2017. 02. 16.", "caseNo": "2016노375", "relateLaword": ["특정범죄 가중처벌 등에 관한 법률(이하  특가법 이라 한다) 제5조", "형법 제331조", "형법 제330조", "형법 제319조", "형사소송법 제364조"], "qotatPrcdnt": ["대법원 2008. 11. 27. 선고2008도7820 판결", "대법원 2015. 10. 15. 선고 2015도8169 판결 "]}, "concerned": {"acusr": "4", "dedat": "1"}, "org": {"orgJdgmnCourtNm": "광주지방법원   ", "orgJdgmnAdjuDe": "2016. 9. 2.", "orgJdgmnCaseNo": "2016고합178"}, "disposal": {"disposalform": "6", "disposalcontent": ["원심판결을 파기한다.", "피고인을 징역 4년에 처한다."]}, "mentionedItems": {"rqestObjet": ["1. 피고인은 이 사건 당시 병적 도벽증상 등으로 심신미약 상태에 있었고(심신미약), 2. 원심의 형(징역 4년)은 너무 무거워서 부당하다(양형부당)", "피고인의 주거침입행위는 특정범죄 가중처벌 등에 관한 법률(이하 특가법이라 한다) 제5조의4 제6항에서 규정하는 상습절도죄에흡수되지 않고, 별개로 주거침입죄가 인정되어 특가법상 상습절도죄와는 실체적 경합관계에 있어야 한다.", "그럼에도 이 부분 공소사실을 무죄로 판단한 원심판결에는 법리오해의 위법이 있다.", "원심의 형은 너무 가벼워서 부당하다."]}, "assrs": {"acusrAssrs": ["피고인의 주거침입행위는 특정범죄 가중처벌 등에 관한 법률(이하 ‘특가법’이라 한다)제5조의4 제6항에서 규정하는 상습절도죄에 흡수되지 않고, 별개로 주거침입죄가 인정되어 특가법상 상습절도죄와는 실체적 경합관계에 있어야 하는데, 그럼에도 이 부분 공소사실을 무죄로 판단한 원심판결에는 법리오해의 위법이 있다.", "원심의 형은 너무 가벼워서 부당하다."], "dedatAssrs": ["피고인은 이 사건 당시 병적 도벽증상 등으로 심신미약 상태에 있었고(심신미약), 원심의 형(징역 4년)은 너무 무거워서 부당하다(양형부당)."]}, "facts": {"bsisFacts": ["피고인은 이 사건 당시 병적 도벽증상 등으로 심신미약 상태에 있었다.", "주간에 주거에 침입하여 절도의 범행을 저질렀다."]}, "dcss": {"courtDcss": ["구 특가법 제5조의4 제5항은 상습성의 요건 외에도 범죄경력과 누범가중에 해당함을 요건으로 하는 반면 구 특가법 제5조의4 제1항은 상습성을 요건으로 하고 있어 그 요건이 서로 다르다.", "형법 제330조에 규정된 야간주거침입절도죄 및 같은 법 제331조 제1항에 규정된 손괴특수절도죄를 제외하고 일반적으로 주거침입은 절도죄의 구성요건이 아니므로 절도범이 그 범행수단으로 주거침입을 한 경우에 그 주거침입행위는 절도죄에 흡수되지 아니하고 별개로 주거침입죄를 구성하여 절도죄와는 실체적 경합의 관계에 서는 것이 원칙이다.", "(대법원 2008.11.27.선고 2008도7820 판결, 대법원 2015.10.15.선고 2015도8169 판결 등 참조)", "따라서 주간에 주거에 침입하여 절도함으로써 특가법 제5조의4 제6항 위반죄가 성립하는 경우에도, 같은 취지에서 별도로 형법 제319조 의 주거침입죄를 구성한다고 봄이 상당하다.", "위와 같은 법리에 비추어 보면, 주간에 주거에 침입하여 절도의 범행을 저지른 피고인에 대하여 특가법 제5조의4 제6항 위반죄에 주거침입죄가 흡수되어 별개로 주거침입죄를 구성하지 않는다고 판단한 원심판결에는 법리오해의 위법이 있다.", "원심 및 당심이 적법하게 채택하여 조사한 증거들에 의하면, 이 사건 범행의 경위 및 수법, 범행 전후의 정황 및 피고인의 진술내용 등에 비추어 볼 때, 피고인이 이 사건 범행 당시 병적 도벽증상 등으로 인하여 사물을 변별하거나 의사를 결정할 능력이 미약한 상태에 있었던 것으로는 보이지 않기에, 따라서 피고인의 이 부분 주장은 이유 없다."]}, "close": {"cnclsns": ["원심판결을 파기한다.", "피고인을 징역 4년에 처한다."]}, "_source_path": "TL_1.판결문/02.형사/2017/2016노375.json"}
        print(case_data)
        # raw = call_gpt_with_case(case_data, system_prompt_v1(), json_schema_v1)
        # try:
        #     obj = parse_json_or_raise(raw)
        #     validate_json_schema(obj)
        # except Exception as e:
        #     print("Parsing/Validation Error:", e)
        #     return

        # with open("output.json", "w", encoding="utf-8") as out_file:
        #     json.dump(obj, out_file, ensure_ascii=False, indent=2)
        #     out_file.write("\n")

        # raw = call_gpt_with_case(case_data, system_prompt_v2(), json_schema_v1)
        # try:
        #     obj = parse_json_or_raise(raw)
        #     validate_json_schema(obj)
        # except Exception as e:
        #     print("Parsing/Validation Error:", e)
        #     return

        # with open("output.json", "a", encoding="utf-8") as out_file:
        #     json.dump(obj, out_file, ensure_ascii=False, indent=2)
        #     out_file.write("\n")

        # 서술형(v5) 문제 생성 실행
        # raw = call_gpt_with_case(case_data, system_prompt_v4(), json_schema_v2)
        raw = call_gpt_with_case(case_data, system_prompt_v5(), json_schema_v3)
        try:
            obj = parse_json_or_raise(raw)
            # 다수 문항 생성 방지: 2개 이상 응답 시 첫 문항만 유지
            if isinstance(obj, dict) and isinstance(obj.get("items"), list) and len(obj["items"]) > 1:
                obj["items"] = [obj["items"][0]]
            validate_json_schema(obj)
        except Exception as e:
            print("Parsing/Validation Error:", e)
            return

        with open("output_written.json", "w", encoding="utf-8") as out_file:
            json.dump(obj, out_file, ensure_ascii=False, indent=2)
            out_file.write("\n")


if __name__ == "__main__":
    main()
