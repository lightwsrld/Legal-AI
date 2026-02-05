import os
import glob
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-OCR-2"

pages_dir = "./dataset/image/result_1"

ocr_out_root = "./dataset/image/ocr_results"

merged_md_path = "./dataset/image/merged_result.mmd"

prompt = "<image>\n<|grounding|>Convert the document to markdown. "

base_size = 1024
image_size = 768
crop_mode = True
save_results = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation="flash_attention_2",
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval().cuda().to(torch.bfloat16)

page_images = sorted(glob.glob(os.path.join(pages_dir, "page_*.jpg")))
if not page_images:
    raise FileNotFoundError(f"No page_*.jpg found in: {pages_dir}")

os.makedirs(ocr_out_root, exist_ok=True)

with open(merged_md_path, "w", encoding="utf-8") as f:
    f.write("")  

for idx, image_file in enumerate(tqdm(page_images, desc="OCR pages")):
    page_out_dir = os.path.join(ocr_out_root, f"page_{idx:03d}")
    os.makedirs(page_out_dir, exist_ok=True)

    _ = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        output_path=page_out_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=save_results,
    )

    page_mmd = os.path.join(page_out_dir, "result.mmd")
    if not os.path.exists(page_mmd):
  
        with open(merged_md_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n<!-- page {idx:03d} : result.mmd not found -->\n")
        continue

    with open(page_mmd, "r", encoding="utf-8", errors="replace") as f:
        page_text = f.read().strip()

    with open(merged_md_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n<!-- ========== PAGE {idx:03d} ({os.path.basename(image_file)}) ========== -->\n\n")
        f.write(page_text)
        f.write("\n") 

print(f"merged markdown saved to: {merged_md_path}")
print(f"per-page outputs saved under: {ocr_out_root}")