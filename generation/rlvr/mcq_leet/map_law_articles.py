import json
import re

def extract_major_premise(answer):
    """대전제 부분만 추출"""
    pattern = r'대전제 \(법 규범\):\s*(.*?)(?=\s*소전제 \(사건 사실\):)'
    match = re.search(pattern, answer, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_law_articles(text):
    """텍스트에서 법률명과 조문 추출"""
    pattern = r'(민법|민사소송법|형법|형사소송법|상법|수도법)\s*제(\d+)조'
    matches = re.findall(pattern, text)
    return matches

def load_law_data():
    """법률 데이터 로드"""
    law_files = {
        '민법': './dataset/raw/law_articles/law_articles_minsa.jsonn',
        '민사소송법': './dataset/raw/law_articles/law_articles_minsaso.json',
        '형법': './dataset/raw/law_articles/law_articles_criminal.json',
        '형사소송법': './dataset/raw/law_articles/law_articles_criminalso.json',
        '상법': './dataset/raw/law_articles/law_articles_sang.json',
        '수도법': './dataset/raw/law_articles/law_articles_sudo.json'
    }

    law_data = {}
    for law_name, file_path in law_files.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            law_data[law_name] = json.load(f)

    return law_data

def find_article(law_data, law_name, article_num):
    """특정 법률의 조문 찾기"""
    if law_name not in law_data:
        return None

    articles = law_data[law_name]
    article_pattern = rf'^제{article_num}조'

    for article in articles:
        if re.match(article_pattern, article):
            return f"{law_name} {article}"

    return None

print("법률 데이터 로딩 중...")
law_data = load_law_data()

total_count = 0
matched_count = 0
no_match_count = 0

results = []

with open("./dataset/sft/legal_ko/precedent_major_laws.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            answer = data.get("answer", "")
            major_premise = extract_major_premise(answer)

            total_count += 1

            # 대전제에서 법률 조문 추출
            queries = []
            if major_premise:
                law_articles = extract_law_articles(major_premise)

                for law_name, article_num in law_articles:
                    article_content = find_article(law_data, law_name, article_num)
                    if article_content:
                        queries.append(article_content)

            # query 필드 추가
            if queries:
                data["query"] = queries
                matched_count += 1
            else:
                data["query"] = []
                no_match_count += 1

            results.append(data)

            if total_count % 500 == 0:
                print(f"처리 중: {total_count}건...")

        except Exception as e:
            print(f"오류 발생: {str(e)}")
            continue

output_path = "./dataset/sft/legal_ko/precedent_with_queries.jsonl"
with open(output_path, "w", encoding="utf-8") as f_out:
    for result in results:
        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"\n완료!")
print(f"총 데이터 수: {total_count}건")
print(f"매칭된 건수: {matched_count}건")
print(f"매칭 안 된 건수 (query가 빈 배열): {no_match_count}건")
print(f"저장 경로: {output_path}")

# query가 빈 값인 데이터 확인
empty_query_count = sum(1 for r in results if not r.get("query"))
print(f"\nquery가 빈 배열인 데이터: {empty_query_count}건")
