
#
# ## 모델확인
#
# from safetensors import safe_open
# import torch
#
# # 모델 파일 경로
# file_path = "models/trained_model/model.safetensors"
#
# # safetensors 열기
# with safe_open(file_path, framework="pt", device="cpu") as f:
#     # 모델 안에 저장된 모든 Tensor 키 (레이어 이름)
#     keys = f.keys()
#     print(f"모델에 저장된 텐서 수: {len(keys)}개")
#     print("\n== 텐서 이름들 일부 출력 ==")
#     for key in list(keys)[:10]:  # 앞에 10개만 보기
#         print(f"  - {key}")
#
#     print("\n== 첫 번째 텐서 내용 ==")
#     first_key = keys[0]
#     tensor = f.get_tensor(first_key)
#     print(f"텐서 이름: {first_key}")
#     print(f"Shape: {tensor.shape}")
#     print(f"데이터 일부:\n{tensor.flatten()[:10]}")



## 업로드

from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="models/trained_model",  # 폴더 전체
    repo_id="kijinny/categoryMapping",
    token="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",
    repo_type="model",
    path_in_repo="."  # 루트에 그대로 업로드
)



# ## 다운로드
# from huggingface_hub import hf_hub_download
#
# model_path = hf_hub_download(
#     repo_id="kijinny/categoryMapping",
#     filename="model.safetensors",
#     token="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",   # Access Token
#     cache_dir="./models/trained_model"
# )
#
# print(f"모델 다운로드 완료: {model_path}")
