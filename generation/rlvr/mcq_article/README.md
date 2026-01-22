## mcq_article

기사 기반 MCQ 생성·후처리 관련 파일 입니다. 

`OPENAI_API_KEY` 환경변수가 필요합니다. 

원천 기사 데이터는 `lawtimes_all.csv`를 사용합니다.

### 파일 개요
- **api_call_v2.py**: 기사 1건을 입력으로 GPT에 문제 생성을 요청하는 모듈. 다양한 시스템 프롬프트(최종 프롬프트 v10), 스키마, 파싱/검증 함수 포함. 스크립트 단독 실행 시 샘플 1건을 뽑아 `output.json` 저장.
- **batch_call.py**: `lawtimes_all.csv`에서 구간/샘플을 뽑아 배치로 문제 생성 후 `batch{n}.csv` 저장.
- **create_article_dataset.py**: `batch5.csv`에서 최대 6000행을 골라 `abridged_context + question`을 합쳐 학습용 `article_dataset.csv` 생성.
- **judge_fixed.py**: 배치 파일(`batch{n}.csv`)을 LLM으로 점검·필터링하여 `batch{n}_filtered.csv` 생성.
- **llm_judge.py**: 임의의 `<name>.csv`를 점검·필터링하여 `<name>_filtered.csv` 생성.
- **article_dataset_filtered.csv**, **batch5.csv**: 생성/후처리 산출물.

### 사용법(.py)
- 환경 준비

```bash
pip install openai python-dotenv tqdm pandas
```

- api_call_v2.py: 기사 1건 샘플 문제 생성

```bash
python api_call_v2.py
# output.json 저장
```

- batch_call.py: 배치 생성
  - batch 5 → v10 프롬프트 → batch5.csv

```bash
python batch_call.py --batch 5
```

- create_article_dataset.py: 만든 배치 결과(csv) -> 문제 포맷(csv)으로 변경

```bash
python create_article_dataset.py
# article_dataset.csv 생성
```

- judge_fixed.py: 배치 파일 품질 필터링
  - 입력: `batch{n}.csv`, 출력: `batch{n}_filtered.csv`

```bash
python judge_fixed.py --batch 5 --start-line 1
```

- llm_judge.py: 임의 파일 품질 필터링
  - 입력: `<name>.csv`, 출력: `<name>_filtered.csv`

```bash
python llm_judge.py --name article_dataset --start-line 1
```


