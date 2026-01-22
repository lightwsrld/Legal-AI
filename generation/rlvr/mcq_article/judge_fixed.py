import csv, json, openai
import os
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_judge_prompt(context, question, answers):
    return f"""
당신은 '법률 MCQA 데이터셋 검증 전문가(Legal MCQA Evaluator)'입니다.
다음 문항의 품질을 법리적·논리적으로 종합 평가하세요.

### [지문]
{context}

### [질문]
{question}

### [선택지]
1. {answers[0]}
2. {answers[1]}
3. {answers[2]}
4. {answers[3]}
5. {answers[4]}

다음 기준으로 JSON 형식으로만 평가 결과를 출력하십시오:

## 2. 오답선지의 법리적 타당성과 오류 방향성 (Distractor Quality) - 완화
- 오답이 완전히 무의미하지 않은가? (극단적인 경우만 체크)

## 3. 선택지 간 의미적 중복 여부 (Semantic Redundancy) - 완화
- 선택지가 완전히 동일하지 않은가? (극단적인 경우만 체크)

## 4. 환각 발생 여부 (Hallucination Detection) - 완화
- 존재하지 않는 법리·판례·법조문을 단정적으로 서술했는가? (명백한 환각만 체크)

## 5. 문항 구조 및 일관성 (Structural Coherence) - 완화
- 문항이 완전히 이해 불가능하지 않은가? (극단적인 경우만 체크)

## 6. 난이도 추정 (Difficulty Scoring)
- 1점: 단순 암기형
- 2점: 단일 법리 적용형
- 3점: 사실과 법리 결합형
- 4점: 예외·경계 판단형
- 5점: 고급 응용형

## 7. 추가 검증 항목 (Additional Validation) - 완화
- 정답이 완전히 논리적 근거 없이 등장하지는 않는가? (극단적인 경우만 체크)
- 오답이 ‘너무 명백한 부정형 진술’로 되어 있지 않은가?
- 지문이 단순 사실서술에 그치지 않고 법적 쟁점을 충분히 유도하는가?
- 정답과 오답 간 의미적 거리(semantic distance)가 지나치게 크지 않은가?
- 정답이 지문 내에서 논리적 근거 없이 단독으로 등장하지는 않는가?

**중요: 반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트나 설명은 포함하지 마세요.**

{{
  "validity": "High/Medium/Low",
  "errors": [
    {{"type": "DistractorIssue", "comment": "오답선지의 법리적 문제점"}},
    {{"type": "Overlap", "comment": "선택지 간 의미적 중복"}},
    {{"type": "Hallucination", "comment": "환각 발생 내용"}},
    {{"type": "StructuralIssue", "comment": "구조적 일관성 문제"}},
    {{"type": "SemanticDistance", "comment": "정답-오답 간 의미적 거리 문제"}},
    {{"type": "LogicalGap", "comment": "논리적 근거 부족"}}
  ],
  "difficulty_score": 1,
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
    prompt = llm_judge_prompt(
        row["abridged_context"],
        row["question"],
        [row["answer1"], row["answer2"], row["answer3"], row["answer4"], row["answer5"]],
    )
    
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
    except Exception as e:
        print("⚠️ JSON Parse Error:", e)
        print(f"🔍 전체 응답: {result_text}")
        return None
    return result

def passes_filter(result, filter_reasons=None):
    try:
        if not result:
            return False

        validity = result.get("validity", "")
        difficulty = result.get("difficulty_score", 3)
        recommendation = result.get("recommendation", "")
        errors = result.get("errors", [])
        
        # errors가 None이거나 리스트가 아닌 경우 처리
        if not isinstance(errors, list):
            errors = []

        #0. 기본 초기화
        if filter_reasons is None:
            filter_reasons = {}

        #1. Hallucination → 극단적인 경우만 제거
        try:
            hallucination_issues = [e for e in errors if e and e.get("type") == "Hallucination"]
            for issue in hallucination_issues:
                comment = issue.get("comment", "") if issue else ""
                if comment:
                    comment = comment.lower()
                    if any(k in comment for k in ["완전히", "전혀", "존재하지 않는", "잘못된"]):
                        if filter_reasons: filter_reasons["hallucination"] = filter_reasons.get("hallucination", 0) + 1
                        return False
        except Exception as ex:
            print(f"⚠️ Hallucination 검사 오류: {ex}")

        #1-1. 추가 검증항목: 명백한 부정형 진술 검사
        try:
            for e in errors:
                if e and e.get("type") == "DistractorIssue":
                    comment = e.get("comment", "") if e else ""
                    if comment:
                        comment = comment.lower()
                        if any(k in comment for k in ["명백한 부정형", "너무 명백한", "부정형 진술"]):
                            if filter_reasons: filter_reasons["obvious_negative"] = filter_reasons.get("obvious_negative", 0) + 1
                            return False
        except Exception as ex:
            print(f"⚠️ 명백한 부정형 진술 검사 오류: {ex}")

        #1-2. 추가 검증항목: 단순 사실서술 검사
        try:
            for e in errors:
                if e and e.get("type") == "StructuralIssue":
                    comment = e.get("comment", "") if e else ""
                    if comment:
                        comment = comment.lower()
                        if any(k in comment for k in ["단순 사실서술", "법적 쟁점 부족", "쟁점 유도 부족"]):
                            if filter_reasons: filter_reasons["factual_only"] = filter_reasons.get("factual_only", 0) + 1
                            return False
        except Exception as ex:
            print(f"⚠️ 단순 사실서술 검사 오류: {ex}")

        #2. 위험 점수 기반 누적 평가 (AnswerValidity 제외)
        risk_score = 0
        for e in errors:
            try:
                # AnswerValidity는 위험 점수 계산에서 제외
                if e and e.get("type") == "AnswerValidity":
                    continue
                comment = e.get("comment", "") if e else ""
                if comment:
                    comment = comment.lower()
                    if "치명" in comment:
                        risk_score += 2
                    elif any(k in comment for k in ["전혀", "완전히", "불가능", "심각"]):
                        risk_score += 1
                    elif any(k in comment for k in ["부분적", "경미", "일부"]):
                        risk_score += 0.5
            except Exception as ex:
                print(f"⚠️ 위험 점수 계산 오류: {ex}")
                continue

        if risk_score >= 8:  # 극도로 심각한 문제 누적 시만 제거
            if filter_reasons: filter_reasons["risk_high"] = filter_reasons.get("risk_high", 0) + 1
            return False

        #3. DistractorIssue (오답 품질) — 4개 이상일 때만 제거
        distractor_issues = [e for e in errors if e and e.get("type") == "DistractorIssue"]
        if len(distractor_issues) >= 4:
            severe_distractors = [d for d in distractor_issues if any(k in d.get("comment", "") for k in ["치명", "완전히", "전혀"])]
            if severe_distractors:
                if filter_reasons: filter_reasons["distractor_issues"] = filter_reasons.get("distractor_issues", 0) + 1
                return False

        #4. StructuralIssue (구조 문제) — 거의 필터링하지 않음
        structural_issues = [e for e in errors if e and e.get("type") == "StructuralIssue"]
        for issue in structural_issues:
            comment = issue.get("comment", "") if issue else ""
            if comment:
                comment = comment.lower()
                if any(k in comment for k in ["완전히", "전혀", "불가능"]):
                    if filter_reasons: filter_reasons["structural_issues"] = filter_reasons.get("structural_issues", 0) + 1
                    return False

        #5. SemanticDistance (선택지 의미 거리) — 거의 필터링하지 않음
        semantic_issues = [e for e in errors if e and e.get("type") == "SemanticDistance"]
        for issue in semantic_issues:
            comment = issue.get("comment", "") if issue else ""
            if comment:
                comment = comment.lower()
                if any(k in comment for k in ["완전히 불일치", "전혀 관련없음", "완전히 다름"]):
                    if filter_reasons: filter_reasons["semantic_distance"] = filter_reasons.get("semantic_distance", 0) + 1
                    return False

        #6. LogicalGap (논리적 비약) — 4개 이상일 때만 제거
        logical_issues = [e for e in errors if e and e.get("type") == "LogicalGap"]
        severe_logicals = [l for l in logical_issues if any(k in l.get("comment", "") for k in ["완전", "전혀", "불가능"])]
        if len(severe_logicals) >= 4:
            if filter_reasons: filter_reasons["logical_gap"] = filter_reasons.get("logical_gap", 0) + 1
            return False

        #6-1. 추가 검증항목: 정답의 논리적 근거 부족 검사
        try:
            for e in errors:
                if e and e.get("type") == "LogicalGap":
                    comment = e.get("comment", "") if e else ""
                    if comment:
                        comment = comment.lower()
                        if any(k in comment for k in ["논리적 근거 없이", "단독으로 등장", "근거 부족"]):
                            if filter_reasons: filter_reasons["insufficient_grounds"] = filter_reasons.get("insufficient_grounds", 0) + 1
                            return False
        except Exception as ex:
            print(f"⚠️ 논리적 근거 부족 검사 오류: {ex}")

        #6-2. 추가 검증항목: 의미적 거리 과도 검사
        try:
            semantic_distance_issues = [e for e in errors if e and e.get("type") == "SemanticDistance"]
            for issue in semantic_distance_issues:
                comment = issue.get("comment", "") if issue else ""
                if comment:
                    comment = comment.lower()
                    if any(k in comment for k in ["지나치게 크다", "과도한 거리", "의미적 거리 과도"]):
                        if filter_reasons: filter_reasons["excessive_distance"] = filter_reasons.get("excessive_distance", 0) + 1
                        return False
        except Exception as ex:
            print(f"⚠️ 의미적 거리 과도 검사 오류: {ex}")

        #7. Validity + Recommendation — AnswerValidity 관련 필터링 주석처리
        # error_count = len(errors)
        # if validity.lower() == "low" and recommendation.lower() in ["remove"] and error_count >= 5:
        #     if filter_reasons: filter_reasons["validity_low"] = filter_reasons.get("validity_low", 0) + 1
        #     return False

        #8. 난이도 필터링 제거 (모든 난이도 허용)
        # → 1점, 5점 문제도 통과시켜서 후처리 단계에서 판단

        #9. Soft Filtering 경고 (경계선 문항)
        if 2.5 <= risk_score < 4:
            if filter_reasons: 
                filter_reasons["warn"] = filter_reasons.get("warn", [])
                filter_reasons["warn"].append("manual review recommended")
            # 통과는 시키되 후처리 검토 필요

        #10. 중복 문제 검사 (5개 이상의 중복만 필터링)
        overlap_count = sum(1 for e in errors if e and e.get("type") == "Overlap")
        if overlap_count >= 5:
            if filter_reasons: filter_reasons["overlap"] += 1
            return False

        #11. 통과
        return True
        
    except Exception as ex:
        print(f"⚠️ 필터링 함수 오류: {ex}")
        # 오류 발생 시 안전하게 통과 처리
        return True

def filter_mcqa(input_csv, output_csv, start_line=1):
    with open(input_csv, newline='', encoding='utf-8') as infile, \
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
            "excessive_distance": 0
        }
        
        # tqdm으로 진행도 표시
        for i, row in enumerate(tqdm(rows, desc="문항 평가 중", unit="문항"), start=start_line):
            try:
                tqdm.write(f"🔍 Evaluating Q{i}: {row['question'][:40]}...")
                result = judge_question(row)

                if passes_filter(result, filter_reasons):
                    writer.writerow(row)
                    passed_count += 1
                    tqdm.write(f" PASSED → Output에 추가됨")
                    if result and result.get("detailed_analysis"):
                        analysis = result["detailed_analysis"]
                        tqdm.write(f"   📊 난이도: {result.get('difficulty_score', 'N/A')}점")
                        tqdm.write(f"   📝 종합평가: {analysis.get('overall_assessment', 'N/A')[:50]}...")
                else:
                    filtered_count += 1
                    tqdm.write(f" FILTERED OUT")
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
            except Exception as ex:
                print(f"⚠️ 문항 {i} 처리 오류: {ex}")
                # 오류 발생 시 안전하게 통과 처리
                writer.writerow(row)
                passed_count += 1
                tqdm.write(f" ERROR → 안전하게 통과 처리됨")

    print(f"\n 필터링 완료!")
    print(f"    총 문항: {total_rows}개")
    print(f"    통과: {passed_count}개")
    print(f"    필터링: {filtered_count}개")
    print(f"    결과 파일: '{output_csv}'")
    
    print(f"\n    필터링 이유별 통계:")
    for reason, count in filter_reasons.items():
        if count > 0:
            print(f"   - {reason}: {count}개")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="법률 MCQA 데이터셋 필터링")
    parser.add_argument("--batch", type=int, required=True, help="배치 번호 (1, 2, 3, ...)")
    parser.add_argument("--start-line", type=int, default=1, help="시작 라인 번호 (기본값: 1)")
    args = parser.parse_args()
    
    batch_num = args.batch
    start_line = args.start_line
    input_file = f"batch{batch_num}.csv"
    output_file = f"batch{batch_num}_filtered.csv"
    
    print(f" 배치 {batch_num} 처리 시작")
    print(f" 입력 파일: {input_file}")
    print(f" 출력 파일: {output_file}")
    print(f" 시작 라인: {start_line}")
    
    filter_mcqa(input_file, output_file, start_line)
