CUDA_VISIBLE_DEVICES=0,1 vllm serve \
  ./.cache/huggingface/hub/models--LGAI-EXAONE--EXAONE-4.0-32B/snapshots/1b30fcd477492634a47b30226af40fda4010cd73 \
  --served-model-name exaone4.0-32b \
  --port 8005 \
  --trust-remote-code \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --api-server-count 20 \
  --max-model-len 14000 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 40960

