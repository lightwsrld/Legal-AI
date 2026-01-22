22#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRPO Training Script for Qwen3-4B with LoRA
외부 vLLM 서버를 사용하여 고속 generation 수행
"""

import os
import re
import numpy as np
import torch
import wandb
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

MODEL_ID = "./.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

wandb.init(project="name", name="qwen3_grpo", config={"model": MODEL_ID, "learning_rate": 5e-6})

print(f"Loading model from {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=None,  
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="kernels-community/vllm-flash-attn3",
)

model.config.use_cache = False
model.config.pretraining_tp = 1

def build_final_prompt(input_text: str) -> str:
    return f"""
    A conversation between User and Assistant. The user asks a question, and the assistant solves it.
    The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively.
    <think> reasoning process here </think> <answer> answer here </answer>.
    Answer must be only a single number between 1 and 5.

    User:
    {input_text}

    Assistant:
    """

print(f"System prompt configured")

print("Loading dataset...")

df = pd.read_csv("./dataset/rlvr/RLVR_1205.csv")
print(f"Original dataset size: {len(df)}")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

def format_question_with_answers(row):
    """question과 answer1~5를 결합하여 프롬프트 생성"""
    answers = []
    for i in range(1, 6):
        ans = row.get(f'answer{i}', '')
        if pd.notna(ans) and str(ans).strip():
            answers.append(f"{i}. {ans}")

    prompt_text = row['question'] + "\n" + "\n".join(answers)
    return prompt_text

def extract_hash_answer(text):
    """답안 추출 함수"""
    return text

def preprocess_data(df):
    """DataFrame을 HuggingFace Dataset으로 변환"""
    data = []
    for _, row in df.iterrows():
        prompt_text = format_question_with_answers(row)
        # final_prompt = build_final_prompt(prompt_text)
        solution = str(row['solution']).strip()

        data.append({
            "prompt": [
                {"role": "system", "content": "You are a careful legal assistant who always provides reasoning and grounds for your conclusions. Reasoning in <think></think>, conclusion in <answer></answer> tag. Colusion must be only a single number between 1 and 5"},
                {"role": "user", "content": prompt_text},
            ], 
            "answer": solution,
        })
    return Dataset.from_list(data)

dataset = preprocess_data(train_df)
dataset = dataset.shuffle(seed=42)
dataset_eval = preprocess_data(val_df)
dataset_eval = dataset_eval.shuffle(seed=42)

print(f"Train dataset size: {len(dataset)}")
print(f"Validation dataset size: {len(dataset_eval)}")
print("\nSample data:")
print(f"Prompt: {dataset[0]['prompt']}")
print(f"Answer: {dataset[0]['answer']}")


answer_start = "<answer>"
answer_end = "</answer>"

answer_end_regex = r"</answer>[\s]{0,}" + \
    "(?:" + re.escape(tokenizer.eos_token) + ")?"

match_format = re.compile(
    rf"{answer_start}\s*([1-5])\s*{answer_end_regex}\s*$",
    flags=re.MULTILINE | re.DOTALL
)

match_numbers = re.compile(
    re.escape(answer_start) + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags=re.MULTILINE | re.DOTALL
)

PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5


def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]

        think_match = re.search(r"<think>", response)

        answer_match = re.search(
            r"<answer>\s*([1-5])\s*</answer>\s*$", 
            response, 
            flags=re.DOTALL
        )

        if think_match and answer_match:
            if think_match.start() < answer_match.start():
                scores.append(1.0)
                continue
        
        scores.append(0.0)

    return scores

def reward_thinking_length(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        think_match = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
        
        if think_match:
            think_content = think_match.group(1).strip()
            # 토큰 수 기준
            num_tokens = len(tokenizer.encode(think_content, add_special_tokens=False))
            
            if num_tokens >= 100:
                scores.append(1.0)
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)
    
    return scores

def check_choice_mcqa(prompts, completions, answer, **kwargs):
    scores = []
    for completion, true_answer in zip(completions, answer):
        response = completion[0]["content"]
        m = match_format.search(response)
        if not m:
            scores.append(0.0)   
            continue
        pred = int(m.group(1))  # 1~5 보장
        # true_answer가 문자열이면 정수화
        try:
            gold = int(float(str(true_answer).strip()))
        except:
            scores.append(0.0)
            continue
        scores.append(1.0 if pred == gold else 0.0)
    return scores


# 토큰화 및 길이 필터링
print("Tokenizing dataset...")
tokenized = dataset.map(
    lambda x: {
        "tokens": tokenizer.apply_chat_template(
            x["prompt"], 
            add_generation_prompt=True, 
            tokenize=True,
            enable_thinking=True,  
        )
    },
    batched=True,
)

print("Sample tokenized output:")
print(tokenizer.decode(tokenized[0]["tokens"]))

tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
MAX_LEN = 4096

def filter_by_length(ds, name: str):
    print(f"\nTokenizing {name} dataset...")
    tokenized = ds.map(
        lambda x: {
            "tokens": tokenizer.apply_chat_template(
                x["prompt"],
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=True,  
            )
        },
        batched=True,
    )

    if name == "train":
        print("Sample tokenized output:")
        print(tokenizer.decode(tokenized[0]["tokens"]))

    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

    lengths = np.array(tokenized["L"])
    mask = lengths <= MAX_LEN
    idx = np.where(mask)[0]

    print(f"{name} 총 샘플 수: {len(ds)}")
    print(f"{name} 중 {MAX_LEN} 토큰 이하 샘플 수: {len(idx)}")
    print(f"{name} 필터링 비율: {len(idx) / len(ds):.4f}")

    ds = ds.select(idx)
    print(f"{name} filtered dataset size: {len(ds)}")

    del tokenized
    return ds

dataset = filter_by_length(dataset, "train")
dataset_eval = filter_by_length(dataset_eval, "eval")

max_prompt_length = 4096
max_completion_length = 4096 


training_args = GRPOConfig(

    learning_rate=1e-6,
    # weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    # optim="adamw_torch_8bit",
    optim='paged_adamw_32bit',

    importance_sampling_level="sequence",
    loss_type="grpo",
    # beta=0.01,
    epsilon=3e-4,
    epsilon_high=4e-4,

    # 배치 및 로깅 설정
    logging_steps=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   
    num_generations=4,

    # 길이 설정
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    
    # 훈련 스텝
    num_train_epochs=1,  
    max_steps=-1,
    save_strategy="steps",  
    save_steps=100,
    output_dir="outputs_grpo_1205_llama_base",
    bf16=True,

    # reward / 수치 안정화
    scale_rewards="batch",              # LitePPO 스타일 batch std 정규화
    # cast_lm_head_to_fp32=True,         # FP32 logits
    mask_truncated_completions=True,   # truncation 안정성

    # eval
    eval_strategy="steps", 
    eval_steps=100,
    
    # WandB 설정
    report_to="wandb",
    run_name=wandb.run.name,
    logging_dir="./logs",

    generation_kwargs={
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.07,
        "max_tokens": max_completion_length, 
        "stop_token_ids": [tokenizer.eos_token_id],
    },

    # Qwen3 thinking 모드
    # chat_template_kwargs={"enable_thinking": True},
)

# vLLM 설정 (colocate 모드, TP=1로 DP 활용)
training_args.use_vllm = True
training_args.vllm_mode = "colocate"
training_args.vllm_tensor_parallel_size = 1         
training_args.vllm_gpu_memory_utilization = 0.5
training_args.vllm_max_model_len = 13000


print("Initializing trainer...")
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        match_format_exactly,
        reward_thinking_length,
        check_choice_mcqa,
        # check_numbers,
    ],
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset_eval, 
)

# 훈련 시작
print("Starting training...")
trainer.train()

# 모델 저장
print("Saving model...")
trainer.save_model("./output/final_model")
tokenizer.save_pretrained("./output/final_model")

# WandB 종료
wandb.finish()

print("Training completed!")
