import csv, json, openai
import re
import os
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_judge_prompt(question, answers, solution):
    return f"""
당신은 '법률 MCQA 데이터셋 검증 전문가(Legal MCQA Evaluator)'입니다.
다음 문항의 품질을 법리적·논리적으로 종합 평가하세요.

### [질문]
{question}

### [선택지]
1. {answers[0]}
2. {answers[1]}
3. {answers[2]}
4. {answers[3]}
5. {answers[4]}

### [정답]
{solution}

정답이 제공되었으므로, 아래 평가 전반에서 제공된 정답을 기준(anchor)으로 삼아 분석하고 오답과의 구별이 명확한지를 확인하십시오.

다음 기준으로 JSON 형식으로만 평가 결과를 출력하십시오:

## 1. 선택지 간 의미적 중복 여부 (Semantic Redundancy)
- 완전 동일에 준하는 중복만 기록

## 2. 환각 발생 여부 (Hallucination Detection)
다음 기준일 때만 '환각'으로 판정하며, 단순 오답과 엄격히 구분합니다.

환각 판정은 "명백한 허구 근거"가 인용될 때에만 허용합니다. 근거를 정확히 제시할 수 없으면 환각으로 기록하지 않습니다.

- 환각의 정의(허구 창작):
  - (a) 실제 존재하지 않는 조항번호·법 조문·판례명을 창작해 단정적으로 제시
  - (b) 해당 분야에서 승인된 법리·판례와 정면으로 충돌하는 '허구적 결론'을 사실처럼 단정(단, 단순한 오해·오판·다수설/소수설 대립은 여기에 해당하지 않음)
  - (c) 법조계에서 통용되지 않는 새로운 법리/용어를 창작하여 사실처럼 단정

- 환각 아님(오답/다른 유형):
  - 실제 존재하는 법령·판례에 대한 오해·과장·부분적 틀림, 판례 태도의 단순 오판·잘못된 일반화
  - 정답이 아닌 선택지, 논증이 부족한 주장, 논리적 비약은 'SemanticDistance' 또는 'LogicalGap'으로만 고려하거나 필요 시 미기록

- 환각 증거 제시 의무(Evidence Rule): 'Hallucination' 유형을 기록하려면 comment에 다음 중 하나를 반드시 포함하세요.
  - [허구 요소 인용] 정확한 허구 조항번호/판례명/용어(예: "형법 제999조"와 같이 존재하지 않는 조문, 실재하지 않는 사건번호·판례명)
  - [직접 모순 인용] 특정 확립 판례(사건번호/선고연월일/법원) 또는 법 조문과 어긋나는 문구를 원문 수준에서 인용한 뒤, 왜 '허구적 결론'에 해당하는지 단정적으로 설명
  - 위를 comment에 정확히 인용할 수 없으면 'Hallucination'으로 기록하지 말고, 필요 시 다른 유형으로만 기록

주의: 위 (a)~(c) 요건에 해당하지 않으면 '환각'으로 기록하지 마세요. 특히 변호사시험 문항에서 해석상 틀림·논증 부족·과장된 진술은 환각이 아닙니다.

## 3. 문항 구조 및 일관성 (Structural Coherence)
- 문항이 사실상 이해 불가능한 수준일 때만 기록
- 문항 유형 주의: ㄱ, ㄴ, ㄷ 등의 진술 조합(예: "ㄱ, ㄴ" / "ㄱ, ㄷ, ㄹ")을 정답으로 고르는 복수정답형일 수 있음
- 복수정답형·조합형이라는 이유만으로 구조 문제로 기록하지 말 것(선지 결합 규칙이 명확하면 정상 구조)

## 4. 추가 검증 항목 (Additional Validation)
- ‘너무 명백한 부정형 진술’만 기록
- ‘단순 사실서술’로 법적 쟁점을 유도하지 못하는 경우만 기록
- 정답과 오답 간 의미적 거리가 지나치게 큰 경우만 기록
- 정답이 지문 내에서 논리적 근거 없이 단독으로 등장하는 경우만 기록

## SemanticDistance 평가 가이드
- 오답-오답 비교 금지: SemanticDistance는 반드시 ‘정답(앵커) vs 선택지’ 기준으로만 판단합니다. 오답 간 비교 결과는 기록하지 않습니다.
- 기록 조건(정답 기준 전용): 아래 3축 중 2개 이상이 ‘명확’하게 상이/상반일 때에만 SemanticDistance를 기록합니다.
  - 법리 범주 상이(예: 증거능력 vs 증명력)
  - 핵심 요건/전제 상이(예: 반대신문권 보장 여부, 구성요건 충족요소)
  - 결론·효과 상반(예: 허용 vs 금지, 인정 vs 불인정)
- 비기록 원칙: 동일 법리 내 경미한 서술·수식·예외조건 차이는 기록하지 않습니다. 2개 미만 충족 시 SemanticDistance 미기록(필요하면 LogicalGap로만 기록).
- 코멘트 요건(완화): 선택지 번호만 명시하고, 차이의 ‘축(법리/전제/결론)’을 1문장으로 요약하세요. 직접 인용은 선택사항입니다.
- 경계선 처리: 경계선 사례는 SemanticDistance로 기록하지 말고, 근거 부족 문제는 LogicalGap으로만 검토하거나 미기록합니다.

중요 지침:
- 에러는 확실히 단정할 수 있는 경우에만 기록하세요. "같음/보임/추정/가능성/의심" 등 추정 표현은 금지합니다.
- errors 배열에는 실제로 해당되는 유형만 포함하세요. 해당되지 않으면 그 유형의 항목 자체를 넣지 마세요.
- 각 에러 객체의 "comment" 키는 구체적이고 확정적인 근거가 있을 때만 쓰세요. "근거 부족", "거리 과도"와 같은 포괄 표현만 있는 코멘트는 금지합니다. 근거가 불충분하면 (type,comment) 쌍 자체를 생략하세요.
- 같은 type의 에러는 여러 개 생성하지 말고 반드시 1개로 통합하세요. 여러 근거가 있으면 해당 type의 단일 객체의 comment에 간결히 병기하세요.

좋은 comment 예시:
- LogicalGap: [선택지 ㄴ] [인용] "증거능력이 없다" → [연쇄누락] (자백보강 필요성)→(형소법 제312조 제4항 요건 충족 여부) 중 후단 누락; [요구 근거] 형소법 제312조 제4항, 대법원 2015도12345(2016.3.24.); [부족 이유] 반대신문권 보장·특신상태 요건 평가 부재; [개선] "반대신문권 보장 및 특신상태 인정 시에 한해 증거능력 부정이 아니라 제한적으로 인정됨"으로 수정.
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트나 설명은 포함하지 마세요. 오류가 없다면 errors는 빈 배열([])이어야 합니다.

{{
  "validity": "High/Medium/Low",
  "errors": [
    {{"type": "SemanticDistance", "comment": "정답-오답 간 의미적 거리를 단정적으로 기재"}},
    {{"type": "Hallucination", "comment": "구체적인 환각 내용 기재"}}
    {{"type": "StructuralIssue", "comment": "핵심 구조 결함을 단정적으로 기재"}},
    {{"type": "Overlap", "comment": "선택지 간 의미적 중복을 단정적으로 기재"}},
    {{"type": "LogicalGap", "comment": "지문 내 근거 부재를 단정적으로 기재"}}
  ],
  "recommendation": "Keep",
  "detailed_analysis": {{
    "distractor_quality": "오답선지 품질 상세 분석",
    "structural_coherence": "구조적 일관성 상세 분석",
    "semantic_distance": "의미적 거리 분석",
    "overall_assessment": "종합 평가"
  }}
}}
"""

def judge_question(row):
    question = row.get("question") or row.get("\ufeffquestion") or ""
    answers = [
        row.get("answer1", ""),
        row.get("answer2", ""),
        row.get("answer3", ""),
        row.get("answer4", ""),
        row.get("answer5", "")
    ]
    solution = row.get("solution", "") or row.get("\ufeffsolution", "")
    prompt = llm_judge_prompt(question, answers, solution)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    
    try:
        result_text = response.choices[0].message.content.strip()
        
        # JSON 부분만 추출 (```json``` 블록이나 순수 JSON)
        if "```json" in result_text:
            start = result_text.find("```json") + 7
            end = result_text.find("```", start)
            if end != -1:
                result_text = result_text[start:end].strip()
        elif "```" in result_text:
            start = result_text.find("```") + 3
            end = result_text.find("```", start)
            if end != -1:
                result_text = result_text[start:end].strip()
        
        # JSON 시작과 끝 찾기
        if result_text.startswith("{"):
            # {로 시작하는 경우, 마지막 } 찾기
            brace_count = 0
            for i, char in enumerate(result_text):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        result_text = result_text[:i+1]
                        break
        
        result = json.loads(result_text)

        # 중복 type 방지: 같은 type은 1개로 통합
        try:
            errors = result.get("errors", [])
            if isinstance(errors, list):
                merged_by_type = {}
                for err in errors:
                    if not isinstance(err, dict):
                        continue
                    err_type = (err.get("type") or "").strip()
                    if not err_type:
                        continue
                    # 빈 코멘트는 생략 규칙 유지
                    err_comment = (err.get("comment") or "").strip()
                    if err_type not in merged_by_type:
                        merged_by_type[err_type] = {"type": err_type}
                        if err_comment:
                            merged_by_type[err_type]["comment"] = err_comment
                    else:
                        # 기존 comment와 병합(중복 문구 제거)
                        prev = merged_by_type[err_type]
                        comments = []
                        if isinstance(prev.get("comment"), str) and prev["comment"].strip():
                            comments.append(prev["comment"].strip())
                        if err_comment:
                            comments.append(err_comment)
                        # 순서 유지하며 중복 제거
                        seen = set()
                        merged_texts = []
                        for c in comments:
                            if c not in seen:
                                seen.add(c)
                                merged_texts.append(c)
                        if merged_texts:
                            prev["comment"] = "; ".join(merged_texts)
                        else:
                            prev.pop("comment", None)
                result["errors"] = list(merged_by_type.values())
        except Exception as _:
            # 문제가 있어도 파싱 자체는 유지
            pass

        # Hallucination 엄격 검증: 모호하거나 근거 불충분하면 SemanticDistance로 강등
        try:
            def _is_ambiguous(text: str) -> bool:
                if not text:
                    return True
                compact = text.replace(" ", "")
                ambiguous_markers = [
                    "같음", "보임", "듯", "추정", "가능", "가능성", "할수", "할 수", "볼수있", "볼 수 있", "해석에따라", "해석에 따라",
                    "~로볼수", "~로 볼 수", "의견", "추측"
                ]
                return any(m in compact for m in ambiguous_markers)

            def _hallucination_has_required_evidence(text: str) -> bool:
                if not text:
                    return False
                if _is_ambiguous(text):
                    return False
                # 존재하지 않는 조문/용어 창작 주장 + 근접 패턴
                has_article_pattern = bool(re.search(r"제\s*\d+\s*조", text))
                claims_nonexist = any(k in text for k in ["존재하지 않", "실재하지 않", "없는 조문", "허구", "창작", "날조"])
                has_nonexistent_claim = has_article_pattern and claims_nonexist

                # 판례/조문 원문 인용 + 정면 모순 단정
                has_case_anchor = any(k in text for k in ["대법원", "헌법재판소", "선고", "사건번호", "전원합의체", "고등법원"]) or bool(re.search(r"\d{4}\s*\w+\s*\d+", text))
                has_contradiction = any(k in text for k in ["정면으로", "모순", "반함", "배치", "정반대"])
                has_direct_citation_contradiction = has_case_anchor and has_contradiction

                # 단정적 어조(모호 금지) 확인
                has_assertive_tone = any(k in text for k in ["이다", "임", "아니다", "아님", "단정", "명백", "확정"])

                return has_assertive_tone and (has_nonexistent_claim or has_direct_citation_contradiction)

            errs = result.get("errors", [])
            if isinstance(errs, list):
                refined = []
                for e in errs:
                    if not isinstance(e, dict):
                        continue
                    e_type = (e.get("type") or "").strip()
                    comment = (e.get("comment") or "").strip()
                    if e_type == "Hallucination":
                        if not _hallucination_has_required_evidence(comment):
                            # 근거가 약하면 환각으로 기록하지 않고 SemanticDistance로 재분류
                            if comment:
                                refined.append({"type": "SemanticDistance", "comment": comment})
                            # 코멘트가 없으면 아예 생략
                            continue
                    refined.append(e)
                result["errors"] = refined
                # 강등 후 동일 type 중복 제거 및 comment 병합
                try:
                    merged_by_type = {}
                    for err in result.get("errors", []) or []:
                        if not isinstance(err, dict):
                            continue
                        t = (err.get("type") or "").strip()
                        if not t:
                            continue
                        c = (err.get("comment") or "").strip()
                        if t not in merged_by_type:
                            merged_by_type[t] = {"type": t}
                            if c:
                                merged_by_type[t]["comment"] = c
                        else:
                            if c:
                                prev_c = merged_by_type[t].get("comment", "").strip()
                                if prev_c and c and c not in prev_c:
                                    merged_by_type[t]["comment"] = f"{prev_c}; {c}" if prev_c else c
                                elif not prev_c and c:
                                    merged_by_type[t]["comment"] = c
                    result["errors"] = list(merged_by_type.values())
                except Exception:
                    pass
        except Exception:
            pass
    except Exception as e:
        print("⚠️ JSON Parse Error:", e)
        print(f"🔍 전체 응답: {result_text}")
        return None
    return result

# Update scoring logic to reflect scores for all but ambiguous cases

def _compute_weighted_score(errors):
    score = 0.0
    for e in errors or []:
        if not e:
            continue
        e_type = (e.get("type") or "").strip()
        raw_comment = e.get("comment")
        # 코멘트가 없으면 점수화하지 않음
        if not raw_comment:
            continue
        comment = raw_comment.strip().lower()

        # 확정 및 비확정 포함, 애매한 어조는 제외
        compact = comment.replace(" ", "")
        is_ambiguous = any(k in compact for k in ["수있음", "수있다", "수 있을", "보입니다", "없음", "있음", "가능성"])

        # 점수는 애매하지 않은 경우에만 반영
        if is_ambiguous:
            continue

        # 기본 가중치 반영
        if e_type == "Hallucination":
            score += 20
        # elif e_type == "DistractorIssue":
        #     score += 4  # 주석 처리: 오류 탐지(법리 방향성) 점수 반영 비활성화
        elif e_type == "StructuralIssue":
            score += 5
        elif e_type == "SemanticDistance":
            score += 5
        elif e_type == "LogicalGap":
            score += 3
        elif e_type == "Overlap":
            score += 3

        # 심각도 키워드 보정
        if any(k in comment for k in ["치명", "전혀", "완전히", "불가능", "명백", "극단"]):
            score += 3
        elif any(k in comment for k in ["심각", "크다", "과도", "현저"]):
            score += 2
        elif any(k in comment for k in ["부분적", "경미", "일부"]):
            score += 1

        # 특정 항목 가중치 강화
        # if e_type == "DistractorIssue" and any(k in comment for k in ["명백한 부정형", "부정형 진술"]):
        #     score += 5  # 주석 처리: DistractorIssue 가중치 비활성화
        if e_type == "SemanticDistance" and any(k in comment for k in ["완전히 불일치", "전혀 관련없음", "완전히 다름"]):
            score += 5
        if e_type == "LogicalGap" and any(k in comment for k in ["논리적 근거 없이", "단독으로 등장", "근거 부족"]):
            score += 4

        # 확정 어조 보너스
        score += 2
    return score

# Updated passes_filter logic
THRESHOLD_HARD_BLOCK = 15.0  # 환각과 심각한 오류에 대한 즉시 차단 임계값

def passes_filter(result, filter_reasons=None):
    try:
        if not result:
            return True

        validity = result.get("validity", "")
        recommendation = result.get("recommendation", "")
        errors = result.get("errors", [])
        
        # errors가 None이거나 리스트가 아닌 경우 처리
        if not isinstance(errors, list):
            errors = []

        #0. 기본 초기화
        if filter_reasons is None:
            filter_reasons = {}

        # 0-1. 공식 기출 보호를 위한 가중치 기반 조기 판정
        # errors 정규화
        try:
            errors = result.get("errors", [])
            if not isinstance(errors, list):
                errors = []
        except Exception:
            errors = []

        # 명백한 환각은 즉시 차단
        for e in errors:
            if e and e.get("type") == "Hallucination":
                comment = (e.get("comment") or "").lower()
                if any(k in comment for k in ["존재하지 않는", "허구", "날조", "완전히 잘못된"]):
                    if filter_reasons is not None:
                        filter_reasons["hallucination"] = filter_reasons.get("hallucination", 0) + 1
                    return False

        total_score = _compute_weighted_score(errors)
        if total_score >= THRESHOLD_HARD_BLOCK:
            if filter_reasons is not None:
                filter_reasons["hard_block"] = filter_reasons.get("hard_block", 0) + 1
            return False
        if 6.0 <= total_score < THRESHOLD_HARD_BLOCK and filter_reasons is not None:
            filter_reasons["warn"] = filter_reasons.get("warn", [])
            filter_reasons["warn"].append("manual review recommended")
        # 조기 통과
        return True

    except Exception as ex:
        print(f"⚠️ 필터링 함수 오류: {ex}")
        # 오류 발생 시 안전하게 통과 처리
        return True

def filter_mcqa(input_csv, output_csv, start_line=1):
    with open(input_csv, newline='', encoding='utf-8-sig') as infile, \
         open(output_csv, "a", newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        
        # 파일이 비어있을 때만 헤더 쓰기
        outfile.seek(0, 2)  # 파일 끝으로 이동
        if outfile.tell() == 0:  # 파일이 비어있으면
            writer.writeheader()

        # 전체 행 수 계산
        rows = list(reader)
        total_rows = len(rows)
        
        # 시작 라인 처리
        if start_line > 1:
            rows = rows[start_line-1:]  # start_line-1 인덱스부터 시작 (0-based)
            print(f"📝 {start_line}번째 라인부터 처리 시작 (총 {len(rows)}개 문항)")
        
        passed_count = 0
        filtered_count = 0
        filter_reasons = {
            "validity_low": 0,
            "recommendation_remove": 0,
            "difficulty_extreme": 0,
            "hallucination": 0,
            "distractor_issues": 0,
            "structural_issues": 0,
            "semantic_distance": 0,
            "logical_gap": 0,
            "answer_validity": 0,
            "overlap": 0,
            "obvious_negative": 0,
            "factual_only": 0,
            "insufficient_grounds": 0,
            "excessive_distance": 0,
            "hard_block": 0
        }
        
        # 점수 평균 집계를 위한 누적 변수
        sum_total_score = 0.0
        num_total_scored = 0
        sum_passed_score = 0.0
        num_passed_scored = 0
        sum_filtered_score = 0.0
        num_filtered_scored = 0

        # 점수 배열 수집 (요청: 배열 형태로 출력/저장)
        all_scores = []
        passed_scores = []
        filtered_scores = []
        
        # tqdm으로 진행도 표시
        if not filter_reasons:
            filter_reasons = {}

        # 필터링 사유를 output에 저장하는 부분 추가
        filtered_output = []

        # Ensure output.json starts with a valid JSON structure if it's empty or invalid
        if not os.path.exists('output.json') or os.stat('output.json').st_size == 0:
            with open('output.json', 'w', encoding='utf-8') as json_file:
                json.dump([], json_file, ensure_ascii=False, indent=4)
        else:
            # Validate JSON structure
            with open('output.json', 'r+', encoding='utf-8') as json_file:
                try:
                    json.load(json_file)
                except json.JSONDecodeError:
                    json_file.seek(0)
                    json_file.truncate()
                    json.dump([], json_file, ensure_ascii=False, indent=4)

        # Ensure score.json starts with a valid JSON structure if it's empty or invalid
        score_json_path = 'score.json'
        if not os.path.exists(score_json_path) or os.stat(score_json_path).st_size == 0:
            # append 모드로도 파일이 없으면 생성되며, 초기 구조를 기록
            with open(score_json_path, 'a', encoding='utf-8') as json_file:
                json.dump({"all": [], "passed": [], "filtered": []}, json_file, ensure_ascii=False, indent=4)
        else:
            # Validate JSON structure
            with open(score_json_path, 'r+', encoding='utf-8') as json_file:
                try:
                    json.load(json_file)
                except json.JSONDecodeError:
                    json_file.seek(0)
                    json_file.truncate()
                    json.dump({"all": [], "passed": [], "filtered": []}, json_file, ensure_ascii=False, indent=4)

        for i, row in enumerate(tqdm(rows, desc="문항 평가 중", unit="문항"), start=start_line):
            try:
                q_preview = (row.get('question') or row.get('\ufeffquestion') or '')
                tqdm.write(f"🔍 Evaluating Q{i}: {q_preview[:40]}...")
                result = judge_question(row)

                # 점수 계산 (로깅용)
                try:
                    score_for_log = _compute_weighted_score(result.get("errors", [])) if result else 0.0
                except Exception:
                    score_for_log = 0.0
                
                # 전체 평균 집계
                sum_total_score += score_for_log
                num_total_scored += 1
                # 전체 배열 수집
                all_scores.append(score_for_log)

                if passes_filter(result, filter_reasons):
                    writer.writerow(row)
                    passed_count += 1
                    tqdm.write(f" PASSED → Output에 추가됨")
                    tqdm.write(f"   🧮 점수: {score_for_log}")
                    # 통과 평균 집계
                    sum_passed_score += score_for_log
                    num_passed_scored += 1
                    # 통과 배열 수집
                    passed_scores.append(score_for_log)
                    if result and result.get("detailed_analysis"):
                        analysis = result["detailed_analysis"]
                        tqdm.write(f"   📝 종합평가: {analysis.get('overall_assessment', 'N/A')[:50]}...")
                else:
                    filtered_count += 1
                    tqdm.write(f" FILTERED OUT")
                    tqdm.write(f"   🧮 점수: {score_for_log}")
                    # 필터 평균 집계
                    sum_filtered_score += score_for_log
                    num_filtered_scored += 1
                    # 필터 배열 수집
                    filtered_scores.append(score_for_log)
                    if result:
                        errors = result.get("errors", [])
                        if errors:
                            tqdm.write(f"    필터링 사유:")
                            for error in errors[:3]:  # 최대 3개 오류만 표시
                                tqdm.write(f"      - {error.get('type', 'Unknown')}: {error.get('comment', '')[:60]}...")
                        if result.get("recommendation") == "Remove":
                            tqdm.write(f"    권장사항: 제거")
                        elif result.get("validity") == "Low":
                            tqdm.write(f"    타당성: 낮음")

                    # output.json에 필터링 사유 저장
                    filtered_output.append({
                        'question_index': i,
                        'question_preview': q_preview,
                        'errors': errors,
                        'score': score_for_log
                    })
            except Exception as ex:
                print(f"⚠️ 문항 {i} 처리 오류: {ex}")
                writer.writerow(row)
                passed_count += 1
                tqdm.write(f" ERROR → 안전하게 통과 처리됨")

        # 필터링 결과를 JSON 파일로 저장
        # Append data to output.json
        if os.path.exists('output.json'):
            with open('output.json', 'r+', encoding='utf-8') as json_file:
                existing_data = json.load(json_file)
                json_file.seek(0)
                json_file.truncate()
                json.dump(existing_data + filtered_output, json_file, ensure_ascii=False, indent=4)
        else:
            with open('output.json', 'w', encoding='utf-8') as json_file:
                json.dump(filtered_output, json_file, ensure_ascii=False, indent=4)

    print(f"\n 필터링 완료!")
    print(f"    총 문항: {total_rows}개")
    print(f"    통과: {passed_count}개")
    print(f"    필터링: {filtered_count}개")
    print(f"    결과 파일: '{output_csv}'")
    
    # 평균 점수 출력
    try:
        if num_total_scored > 0:
            overall_avg = sum_total_score / max(1, num_total_scored)
            print(f"\n    평균 점수(전체): {overall_avg:.2f} (N={num_total_scored})")
        if num_passed_scored > 0:
            passed_avg = sum_passed_score / max(1, num_passed_scored)
            print(f"    평균 점수(통과): {passed_avg:.2f} (N={num_passed_scored})")
        if num_filtered_scored > 0:
            filtered_avg = sum_filtered_score / max(1, num_filtered_scored)
            print(f"    평균 점수(필터링): {filtered_avg:.2f} (N={num_filtered_scored})")
    except Exception as _:
        pass

    # 점수 배열 출력
    try:
        print("\n    점수 배열(전체):", json.dumps(all_scores, ensure_ascii=False))
        print("    점수 배열(통과):", json.dumps(passed_scores, ensure_ascii=False))
        print("    점수 배열(필터링):", json.dumps(filtered_scores, ensure_ascii=False))
    except Exception as _:
        pass

    # score.json에 누적 저장 (중단 대비: 기존 배열에 이어붙이기)
    try:
        score_payload = {
            "all": all_scores,
            "passed": passed_scores,
            "filtered": filtered_scores,
        }
        if os.path.exists('score.json'):
            with open('score.json', 'r+', encoding='utf-8') as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = {}
                if not isinstance(existing, dict):
                    existing = {}
                existing.setdefault('all', [])
                existing.setdefault('passed', [])
                existing.setdefault('filtered', [])
                # 이어붙이기
                existing['all'].extend(score_payload['all'])
                existing['passed'].extend(score_payload['passed'])
                existing['filtered'].extend(score_payload['filtered'])
                f.seek(0)
                f.truncate()
                json.dump(existing, f, ensure_ascii=False, indent=4)
        else:
            with open('score.json', 'w', encoding='utf-8') as f:
                json.dump(score_payload, f, ensure_ascii=False, indent=4)
    except Exception as _:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="법률 MCQA 데이터셋 필터링")
    parser.add_argument("--name", type=str, required=True, help="파일 이름")
    parser.add_argument("--start-line", type=int, default=1, help="시작 라인 번호 (기본값: 1)")
    args = parser.parse_args()
    
    name = args.name
    start_line = args.start_line
    input_file = f"{name}.csv"
    output_file = f"{name}_filtered.csv"
    
    print(f" 입력 파일: {input_file}")
    print(f" 출력 파일: {output_file}")
    print(f" 시작 라인: {start_line}")
    
    filter_mcqa(input_file, output_file, start_line)
