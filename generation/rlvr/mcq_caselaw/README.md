## mcq_caselaw

판례 기반 MCQ/FITB/서술형 문제 생성·번역·후처리를 위한 파일들입니다. 

`OPENAI_API_KEY` 환경변수가 필요합니다. 

주요 산출물은 `batch*.csv`, `caselaw_dataset_*.csv`, 번역본 CSV 등입니다.

### 파일 개요
- **all_cases.ndjson**: 판례 원천 데이터(NDJSON).
- **api_call_es.py**: 판례 JSON 1건을 불러 GPT로 문제를 생성하는 스크립트. 다양한 시스템 프롬프트(v1~v5)와 스키마, 파싱/검증 유틸 포함. 실행 시 `output_written.json` 생성.
- **batch_call.py**: `all_cases.ndjson`을 배치 단위로 잘라 무작위 샘플을 추출해 문제를 생성하고 `batch{n}.csv`로 저장.
- **create_caselaw_dataset.py**: `batch{1|3|4|5}.csv`를 학습용 데이터셋(`caselaw_dataset_{n}.csv`)으로 정리.
- **csv_to_jsonl.py**: CSV의 `question`/`solution` 컬럼만 추출해 JSONL(`question`, `answer`)로 변환.
- **llm_judge_1.py**, **llm_judge_3.py**: LLM 기반 품질 점검 및 필터링기. 입력 CSV를 읽어 기준에 따라 필터링 결과를 출력.
- **translate_written.py**: `caselaw_written.csv` 내 `solution`을 영문으로 번역. 전체/부분(batch) 처리 지원.
- 그 외 `batch*.csv`, `caselaw_dataset_*.csv`, `caselaw_written*.csv`, `caselaw_fitb.csv`: 생성/후처리 산출물.

### 사용법(.py)
- 환경 준비

```bash
pip install openai python-dotenv tqdm pandas
```

- api_call_es.py: 판례 1건 샘플 문제 생성

```bash
python api_call_es.py
# output_written.json 생성, batch 실행 전 동작 여부, 문제 품질 체크 용
```

- batch_call.py: 배치 생성
  - batch 1: v1 + json_schema_v1 → batch1.csv # 판례 v1(문맥 존제)
  - batch 3: v3 + json_schema_v2 → batch3.csv # 판례 v2(문맥 X)
  - batch 4: v4 + json_schema_v2 → batch4.csv # FITB
  - batch 5: v5 + json_schema_v3 → batch5.csv # 서술형

```bash
python batch_call.py --batch {batch_number}
```

- create_caselaw_dataset.py: 만든 배치 결과(csv) -> 문제 포맷(csv)으로 변경

```bash
python create_caselaw_dataset.py --batch {batch_number} --output {output_filename}.csv
```

- csv_to_jsonl.py: CSV → JSONL(질문/정답만)

```bash
python csv_to_jsonl.py -i {input_filename}.csv -o {output_filename}.jsonl
```

- llm_judge_1.py / llm_judge_3.py: 품질 필터링 (각각 판례 v1, v3)
둘 간의 차이는 문맥과 선지 간의 관련성에 대한 판단
  - 입력: `<name>.csv`, 출력: `<name>_filtered.csv`

```bash
python llm_judge_1.py --name {input.csv} --start-line 1
python llm_judge_3.py --name {output.csv} --start-line 1
```

- translate_written.py: 해설(solution) 영문화
  - 기본 입력: `caselaw_written.csv`
  - `--batch 1|2`로 구간 처리 시 출력 경로 자동 지정

```bash
python translate_written.py --input caselaw_written.csv --output caselaw_written_en.csv 

# 부분 처리 예시(출력 자동 경로: _en_1.csv / _en_2.csv)
python translate_written.py --batch 1
python translate_written.py --batch 2
```


