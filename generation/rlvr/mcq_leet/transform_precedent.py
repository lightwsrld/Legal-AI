import json

input_file = "./dataset/sft/legal_ko/precedent_with_sum.jsonl"
output_file = "./dataset/sft/legal_ko/precedent_transformed.jsonl"

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:

    for line in infile:
        data = json.loads(line.strip())

        # query 리스트를 하나의 문자열로 합침
        query_list = data.get('query', [])
        if isinstance(query_list, list):
            query_combined = '\n\n'.join(query_list)
        else:
            query_combined = query_list

        # 새로운 query 형식으로 변환
        new_query = query_combined + "\n\n위의 내용을 바탕으로 법률 삼단논법을 활용하여 판결을 도출하라."

        # 새로운 데이터 구성 (question 제외, query와 answer만)
        new_data = {
            'query': new_query,
            'answer': data.get('answer', '')
        }

        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

print(f"변환 완료: {output_file}")
