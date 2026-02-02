import os
import requests

API_KEY = ""
URL = "https://api.upstage.ai/v1/document-digitization"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

DATA = {
    "model": "ocr"
}


def merge_lines_into_paragraphs(text):
    lines = text.split('\n')
    paragraphs = []
    buffer = ''

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if buffer:
                paragraphs.append(buffer.strip())
                buffer = ''
        else:
            if buffer:
                buffer += ' ' + stripped
            else:
                buffer = stripped

    if buffer:
        paragraphs.append(buffer.strip())

    return paragraphs

INPUT_DIR = "./dataset/raw/LEET"
TARGET_PDF = "./dataset/raw/LEE/2026_추리논증 (홀).pdf"  

PDF_PATH = os.path.join(INPUT_DIR, TARGET_PDF)

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF 파일이 없습니다: {PDF_PATH}")

print(f"처리 중: {TARGET_PDF}")

with open(PDF_PATH, "rb") as f:
    files = {"document": f}
    response = requests.post(
        URL,
        headers=HEADERS,
        files=files,
        data=DATA
    )

result = response.json()

if result.get("mimeType") == "multipart/form-data":
    texts = [page.get("text", "") for page in result.get("pages", [])]
    joined_text = "\n\n".join(texts)

    merged_paragraphs = merge_lines_into_paragraphs(joined_text)

    output_filename = TARGET_PDF.replace(".pdf", "_cleaned.txt")
    output_path = os.path.join(INPUT_DIR, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        for para in merged_paragraphs:
            f.write(para + "\n\n")

    print(f"→ 저장 완료: {output_filename}")

else:
    print("OCR 처리 실패")
    print("응답 내용:", result)
