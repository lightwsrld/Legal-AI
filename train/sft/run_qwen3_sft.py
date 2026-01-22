#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Qwen3-4B 모델에 LoRA를 적용하여 수학 문제 해결 훈련
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split
from transformers import TrainerCallback  
from tqdm import tqdm
import wandb
from datasets import load_dataset, Dataset
import pandas as pd
import wandb
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_ID = "Qwen/Qwen3-8B"
wandb.init(project="name", name="qwen3_sft", config={"model": MODEL_ID, "learning_rate": 1e-5})

print(f"Loading model from {MODEL_ID}...")

def prepare_dataset(tokenizer, system_prompt, reasoning_start, reasoning_end, 
                   solution_start, solution_end, max_seq_length=4096):
    """데이터셋 준비 및 전처리"""
    print("Loading dataset...")
    
    dataset = pd.read_csv("./dataset/sft/sft_1205_a.csv")

    train_df, val_df = train_test_split(dataset, test_size=0.2, random_state=42)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    def format_dataset(x):
        problem = x["problem"]
        
        thoughts = x["generated_solution"]
        
        final_prompt = f"{reasoning_start}{thoughts}{reasoning_end}{solution_start}{solution_end}"
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": final_prompt}
        ]
    
    dataset["Messages"] = dataset.apply(format_dataset, axis=1)
    
    dataset["N"] = dataset["Messages"].apply(
        lambda x: len(tokenizer.apply_chat_template(x))
    )
    dataset = dataset.loc[dataset["N"] <= max_seq_length].copy()
    print(f"Dataset size after length filtering: {len(dataset)}")
    
    dataset["text"] = tokenizer.apply_chat_template(
        dataset["Messages"].values.tolist(), 
        tokenize=False
    )
    
    train_df, val_df = train_test_split(dataset, test_size=0.1, random_state=42, shuffle=True)
    train_dataset = Dataset.from_pandas(train_df[["text"]], preserve_index=False)
    val_dataset   = Dataset.from_pandas(val_df[["text"]], preserve_index=False)
    
    return train_dataset, val_dataset

def main():
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="kernels-community/vllm-flash-attn3",
        # device_map="auto",
        device_map=None,
    )
    
    
    system_prompt = (
        f"You are given a problem.\n"
        f"Think about the problem and provide your working out.\n"
        f"Provide step-by-step reasoning followed by your final answer.\n"
    )
    
    print(f"System prompt:\n{system_prompt}\n")
    

    max_length = 8192
    train_dataset = Dataset.from_parquet("./dataset/sft/train_dataset_1205_a.parquet")
    eval_dataset = Dataset.from_parquet("./dataset/sft/val_dataset_1205_a.parquet")

    training_args = SFTConfig(
        output_dir="./output",
        dataset_text_field="text",
        max_length=max_length,         
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        # warmup_=5,
        warmup_ratio=0.05,
        num_train_epochs=3,
        learning_rate=1e-5,
        eval_strategy="epoch", 
        # eval_steps=30,

        # 로깅 
        logging_steps=5,
        # 체크포인트 저장 설정
        save_strategy="epoch",  
        # save_steps=200, 
        save_total_limit=20, 

        # 최적화 
        # optim="adamw/_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        gradient_checkpointing=False,
        # ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        packing=True,
        optim = "paged_adamw_32bit",

        # WandB 설정
        run_name=wandb.run.name,
        report_to="wandb",
        logging_dir="./logs",
        torch_empty_cache_steps=4,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    
    # 훈련 
    print("Starting training...")
    trainer.train()
    
    # 모델 
    print("Training completed!")
    trainer.save_model("./output_qwen/final_model")
    tokenizer.save_pretrained("./output_qwen/final_model")
    print("Model saved to ./output_qwen/final_model")


if __name__ == "__main__":
    main()