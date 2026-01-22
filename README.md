## 환경 준비

```bash
conda env create -f environment.yaml
conda activate trl
```

## A100 GPU 8장 기준

### SFT 학습 실행

bash ./train/sft/run_sft.sh

### RLVR 학습 실행

bash ./train/rlvr/run_GSPO_batch.sh

