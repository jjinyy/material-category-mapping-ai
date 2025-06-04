# scripts/upload_model.py
from huggingface_hub import HfApi
# import os

# 환경변수에서 토큰 가져오기
# token = os.getenv("HF_TOKEN")
token="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"

# HfApi 인스턴스 생성
api = HfApi()

# 업로드할 폴더 (trained_model 폴더 전체)
folder_path = "models/trained_model"

# Hugging Face 리포지토리
repo_id = "jjinny/categoryMapping"

# 업로드 실행
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    token=token,
    repo_type="model",
    path_in_repo=".",         # 리포지토리 루트에 업로드
    overwrite=True            # 덮어쓰기 (항상 최신 모델로)
)

print(f"모델 업로드 완료: {repo_id}")
