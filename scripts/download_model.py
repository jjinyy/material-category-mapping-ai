# scripts/download_model.py

from huggingface_hub import snapshot_download
# import os

# 환경변수에서 토큰 가져오기
# token = os.getenv("HF_TOKEN")
token="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"

# Hugging Face 리포지토리
repo_id = "jjinny/categoryMapping"

# 다운로드
snapshot_download(
    repo_id=repo_id,
    local_dir="models/trained_model",
    use_auth_token=token,
    repo_type="model",
    local_dir_use_symlinks=False
)

print(f"모델 다운로드 완료: {repo_id}")
