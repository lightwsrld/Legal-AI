import os
from huggingface_hub import snapshot_download

HUGGING_FACE_TOKEN = ""

repo_id = "joonhok-exo-ai/korean_law_open_data_precedents"  
local_folder = "./Legal/dataset/raw"   
cache_folder = "./cache"                 

print(f"'{repo_id}' 데이터셋 다운로드를 시작합니다.")
print(f"저장될 위치: {os.path.abspath(local_folder)}")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_folder,
    cache_dir=cache_folder,
    local_dir_use_symlinks=False, 
                                  
    token=HUGGING_FACE_TOKEN,
)

print("\n다운로드가 완료되었습니다!")
print(f"이제 '{local_folder}' 폴더에서 파일들을 확인할 수 있습니다.")

