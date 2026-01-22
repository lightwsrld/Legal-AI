import csv
import json
import argparse # argparse 모듈 임포트

def convert_csv_to_jsonl_qs(csv_file_path, jsonl_file_path):
    """
    CSV 파일에서 'question'과 'solution' 필드만 추출하여 JSONL 파일로 변환합니다.

    CSV의 첫 번째 행은 헤더(key)로 사용됩니다.
    'solution' 필드는 정수(int)로 변환을 시도합니다.
    'question' 또는 'solution'이 비어있는 행은 건너뜁니다.
    """
    try:
        with open(csv_file_path, mode='r', encoding='utf-8-sig') as csv_file:
            # CSV 파일을 딕셔너리 형태로 읽어들입니다.
            reader = csv.DictReader(csv_file)

            with open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
                for row in reader:
                    extracted_data = {}

                    # 'question'과 'solution' 필드가 있는지, 그리고 비어있지 않은지 확인
                    if 'question' in row and row['question'] and 'solution' in row and row['solution']:
                        
                        # 1. 'question' 필드 추가
                        extracted_data['question'] = row['question']
                        
                        # 2. 'solution' 필드 처리
                        solution_val = row['solution']
                        try:
                            # 'solution' 값이 숫자인 경우 정수로 변환 시도
                            extracted_data['answer'] = int(solution_val)
                        except (ValueError, TypeError):
                            # solution 값이 숫자가 아닌 경우 원본 문자열 유지
                            extracted_data['answer'] = solution_val
                            
                        # 딕셔너리를 JSON 문자열로 변환 (ensure_ascii=False로 한글 유지)
                        json_line = json.dumps(extracted_data, ensure_ascii=False)
                        
                        # JSONL 파일에 한 줄씩 씁니다.
                        jsonl_file.write(json_line + '\n')
                    
                    # else: 'question' 또는 'solution'이 없거나 비어있으면 해당 행은 무시합니다.

        print(f"성공: '{csv_file_path}'에서 'question'과 'solution'을(를) 추출하여 '{jsonl_file_path}'(으)로 변환하였습니다.")

    except FileNotFoundError:
        print(f"오류: '{csv_file_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

# --- 실행 ---
# argparse를 사용하여 커맨드 라인 인수를 처리하는 부분
def main():
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description="CSV 파일의 'question'과 'solution'을 JSONL로 변환합니다.")
    
    # 입력 파일 인자 추가 (-i 또는 --input)
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        required=True, 
        help='입력 CSV 파일 경로 (예: article_dataset_swapped.csv)'
    )
    
    # 출력 파일 인자 추가 (-o 또는 --output)
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        required=True, 
        help='출력 JSONL 파일 경로 (예: article_qs_only.jsonl)'
    )
    
    # 인자 파싱
    args = parser.parse_args()
    
    # 변환 함수 실행 (파싱된 인자 사용)
    convert_csv_to_jsonl_qs(args.input, args.output)

# 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()