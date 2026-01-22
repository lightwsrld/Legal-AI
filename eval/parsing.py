import pandas as pd
import os

def parse_model_response(raw_response: str) -> dict:
    """
    모델의 전체 응답 문자열을 '</think>' 태그 기준으로 파싱
    
    :param raw_response: 모델이 생성한 원본 문자열.
    :return: 'thinking'과 'assistant' 키를 가진 딕셔너리.
    """
    parsing_delimiter = "</think>"
    
    if parsing_delimiter in raw_response:
        thinking_part, assistant_part = raw_response.split(parsing_delimiter, 1)
        return {
            "thinking": thinking_part.strip(),
            "assistant": assistant_part.strip()
        }
    else:
        return {
            "thinking": "",  
            "assistant": raw_response.strip()
        }

def load_prompts_with_indices(file_path: str) -> list:
    """단일 CSV 파일에서 프롬프트를 로드하고, 추적을 위해 행 인덱스를 함께 반환합니다."""
    prompts_with_indices = []
    if os.path.exists(file_path) and file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        for idx, row in df.iterrows():
            prompt_parts = []
            if pd.notna(row.get("지문")):
                prompt_parts.append(f"[passage]\n{row['지문']}")
            if pd.notna(row.get("질문")):
                prompt_parts.append(f"[question]\n{row['질문']}")
            if pd.notna(row.get("보기")):
                prompt_parts.append(f"[list of statements]\n{row['보기']}")
            if pd.notna(row.get("선택지")):
                prompt_parts.append(f"[option]\n{row['선택지']}")

            prompt = "\n\n".join(prompt_parts)
            if prompt:
                prompts_with_indices.append((idx, prompt))
    return prompts_with_indices