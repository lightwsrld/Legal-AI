import pandas as pd
import os

def create_article_dataset():
    """
    batch5.csv에서 데이터를 추출하여 article_dataset.csv를 생성합니다.
    - abridged_context + question을 합쳐서 question 열로 변환
    - answer1,2,3,4,5는 그대로 가져오기
    - correct_answer는 solution으로 변경
    - 총 2000문제 추출
    """
    
    batch5_path = './generation/rlvr/mcq_article/batch5.csv'
    output_path = './generation/rlvr/mcq_article/article_dataset.csv'
    
    print("batch5.csv 파일을 읽는 중...")
    df_batch5 = pd.read_csv(batch5_path)
    print(f"batch5.csv 총 행 수: {len(df_batch5)}")
    
    # batch5에서 2000행 추출 (또는 전체가 2000개 미만이면 전체)
    batch5_sample = df_batch5.head(6000)
    
    # batch5 데이터 변환: abridged_context + question을 합쳐서 question으로
    article_processed = pd.DataFrame()
    article_processed['question'] = batch5_sample['abridged_context'] + " " + batch5_sample['question']
    article_processed['answer1'] = batch5_sample['answer1']
    article_processed['answer2'] = batch5_sample['answer2']
    article_processed['answer3'] = batch5_sample['answer3']
    article_processed['answer4'] = batch5_sample['answer4']
    article_processed['answer5'] = batch5_sample['answer5']
    article_processed['solution'] = batch5_sample['correct_answer']
    
    print(f"처리된 행 수: {len(article_processed)}")
    
    article_processed.to_csv(output_path, index=False, encoding='utf-8')
    print(f"article_dataset.csv 파일이 생성되었습니다: {output_path}")
    
    print("\n=== 처리 결과 요약 ===")
    print(f"batch5에서 추출한 문제 수: {len(article_processed)}")
    print(f"출력 파일: {output_path}")
    
    return article_processed

if __name__ == "__main__":
    dataset = create_article_dataset()

