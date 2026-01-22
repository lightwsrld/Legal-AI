import pandas as pd
import os
import argparse

EXPECTED_COLUMNS = [
    "batch_number",
    "number",
    "caseNm",
    "caseNo",
    "courtNm",
    "abridged_context",
    "question",
    "answer1",
    "answer2",
    "answer3",
    "answer4",
    "answer5",
    "correct_answer",
    "explanation",
]


def build_dataset_for_batch(batch_csv_path: str, batch_number: int) -> pd.DataFrame:
    print(f"batch{batch_number}.csv 파일을 읽는 중...")
    # batch5.csv 등에 불필요한 추가 열이 붙은 경우를 대비하여 필요한 열만 읽어온다.
    df = pd.read_csv(
        batch_csv_path,
        usecols=lambda c: c in EXPECTED_COLUMNS,
        engine="python",  # 열 개수가 가변적인 경우에도 파서를 중단시키지 않도록 python 엔진 사용
        on_bad_lines="skip",  # 잘못된 행은 건너뛰기
    )
    print(f"batch{batch_number}.csv 총 행 수: {len(df)}")

    sample = df.head(3000)

    if batch_number == 1:
        processed = pd.DataFrame()
        processed['question'] = sample['abridged_context'] + " " + sample['question']
        processed['answer1'] = sample['answer1']
        processed['answer2'] = sample['answer2']
        processed['answer3'] = sample['answer3']
        processed['answer4'] = sample['answer4']
        processed['answer5'] = sample['answer5']
        processed['solution'] = sample['correct_answer']
    elif batch_number == 3:
        processed = pd.DataFrame()
        processed['question'] = sample['question']
        processed['answer1'] = sample['answer1']
        processed['answer2'] = sample['answer2']
        processed['answer3'] = sample['answer3']
        processed['answer4'] = sample['answer4']
        processed['answer5'] = sample['answer5']
        processed['solution'] = sample['correct_answer']
    elif batch_number == 4 or batch_number == 5:
        processed = pd.DataFrame()
        processed['question'] = sample['question']
        processed['answer1'] = sample['answer1']
        processed['answer2'] = sample['answer2']
        processed['answer3'] = sample['answer3']
        processed['answer4'] = sample['answer4']
        processed['answer5'] = sample['answer5']
        processed['solution'] = sample['correct_answer']
    else:
        raise ValueError("지원되지 않는 배치 번호입니다. 1, 3, 4, 5만 허용됩니다.")

    print(f"batch{batch_number}에서 처리된 행 수: {len(processed)}")
    return processed


def create_caselaw_dataset(batch: int, output_path: str | None = None):
    """
    지정한 배치(1 또는 3)만 처리하여 데이터셋을 생성합니다.
    - batch1.csv: abridged_context + question을 합쳐서 question 열로 변환
    - batch3.csv: question 열 그대로 사용
    - 각 batch에서 최대 4500행 사용
    """

    batch1_path = './generation/rlvr/mcq_caselaw/batch1.csv'
    batch3_path = './generation/rlvr/mcq_caselawbatch3.csv'
    batch4_path = './generation/rlvr/mcq_caselaw/batch4.csv'
    batch5_path = './generation/rlvr/mcq_caselaw/batch5.csv'

    if batch == 1:
        final_dataset = build_dataset_for_batch(batch1_path, 1)
    elif batch == 3:
        final_dataset = build_dataset_for_batch(batch3_path, 3)
    elif batch == 4:
        final_dataset = build_dataset_for_batch(batch4_path, 4)
    elif batch == 5:
        final_dataset = build_dataset_for_batch(batch5_path, 5)
    else:
        raise ValueError("지원되지 않는 배치 번호입니다. 1, 3, 4, 5만 허용됩니다.")

    # 출력 경로 결정
    if not output_path:
        suffix = '1' if batch == 1 else ('3' if batch == 3 else ('4' if batch == 4 else '5'))
        output_path = f'./generation/rlvr/mcq_caselaw/caselaw_dataset_{suffix}.csv'

    # 저장
    final_dataset.to_csv(output_path, index=False, encoding='utf-8')
    print(f"출력 파일 생성: {output_path}")

    # 요약
    print("\n=== 처리 결과 요약 ===")
    print(f"batch{batch} 처리")
    print(f"총 문제 수: {len(final_dataset)}")
    print(f"출력 파일: {output_path}")

    return final_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="caselaw 데이터셋 생성기")
    parser.add_argument('--batch', type=int, choices=[1, 3, 4, 5], required=True, help='처리할 배치 번호 (예: --batch 1, --batch 3, --batch 4, --batch 5)')
    parser.add_argument('--output', type=str, default=None, help='출력 파일 경로 (옵션). 미지정 시 caselaw_dataset_1.csv 또는 caselaw_dataset_3.csv')
    args = parser.parse_args()

    create_caselaw_dataset(args.batch, args.output)
