import json
import yaml
import csv
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split 

CONFIG_PATH = "./train/sft/data.yaml"
TRAINSET_PATH = "./dataset/sft/sft_1205_a.csv"  
OUTPUT_TRAIN = "./dataset/sft/train_dataset_1205_a.parquet"
OUTPUT_VAL   = "./dataset/sft/val_dataset_1205_a.parquet"

MAX_SEQ_LENGTH = 8192


def build_messages(question, answer):
    return [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer.strip()},
    ]


def load_data_csv(path):
    dataset = []
    with open(path, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row.get("problem")
            answer = row.get("generated_solution")
            if not question or not answer:
                continue
            dataset.append({"messages": build_messages(question, answer)})
    return dataset


# 1. Config 로드
print(f"Using config file at: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)
print(f"Config: {config}")

# 2. 토크나이저 로드
model_name = config["model"]
print(f"Loading tokenizer from: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 3. Chat template 적용
if config.get("chat_template"):
    tokenizer.chat_template = config["chat_template"]

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 5. EOS/PAD 설정
if "<|im_end|>" in tokenizer.get_vocab():
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|im_end|>"
    print("Set eos_token=<|im_end|>")

tokenizer.padding_side = "right"

# 6. CSV 로드
raw_data = load_data_csv(TRAINSET_PATH)
print(f"로드된 데이터 수: {len(raw_data)}")


train_data, val_data = train_test_split(
    raw_data,
    test_size=0.1,
    random_state=42,
    shuffle=True,
)

print(f"Split 후 크기: train={len(train_data)}, val={len(val_data)}")


def process_dataset(data_list):
    ds = Dataset.from_list(data_list)

    def compute_length(example):
        ids = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            add_generation_prompt=False,
        )
        return {"length": len(ids)}

    ds = ds.map(compute_length)

    before = len(ds)
    ds = ds.filter(lambda x: x["length"] <= MAX_SEQ_LENGTH)
    after = len(ds)
    print(f"길이 제한 필터링: {before} → {after}")

    def apply_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(apply_template, remove_columns=["messages", "length"])
    return ds


# 8. train / val 각각 처리
train_dataset = process_dataset(train_data)
val_dataset = process_dataset(val_data)

# 9. Parquet로 각각 저장
train_dataset.to_parquet(OUTPUT_TRAIN)
val_dataset.to_parquet(OUTPUT_VAL)

print(f"\nTrain 저장: {OUTPUT_TRAIN}")
print(f"Val   저장: {OUTPUT_VAL}")
