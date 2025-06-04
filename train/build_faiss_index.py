"""
build_faiss_index.py
--------------------

학습된 문장 임베딩 모델과 카테고리 데이터를 기반으로
FAISS 인덱스를 구축하고, 관련 메타데이터 저장

1. 모델 로딩 (기존 fine-tuned 모델이 있으면 로딩)
2. 카테고리 데이터(category.csv) 로드
3. 한글/영어 카테고리 텍스트 파싱
4. 카테고리 라벨 임베딩 생성
5. FAISS 인덱스 생성 및 벡터 추가
6. FAISS 인덱스 파일 저장 (models/faiss_index.bin)
7. 인덱스 메타데이터 CSV 저장 (models/faiss_metadata.csv)
8. 인덱스 ID ➔ 카테고리 매핑 JSON 저장 (models/faiss_mapping.json)

"""

import pandas as pd
import numpy as np
import faiss
import os
import json
from sentence_transformers import SentenceTransformer

# 1. 모델 로딩 (파인튜닝 모델이 있으면 그것 사용)
MODEL_PATH = "models/trained_model"
if os.path.exists(MODEL_PATH):
    print("파인튜닝된 모델 로드")
    model = SentenceTransformer(MODEL_PATH)
else:
    print("기본 모델 로드 (all-mpnet-base-v2)")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 2. 카테고리 CSV 로드
csv_path = "data/category.csv"
df = pd.read_csv(csv_path, encoding="ISO-8859-1")  # 필요에 따라 encoding 변경

# 3. 영어/한글 카테고리 모두 처리
all_categories = []
for lang in ["ENG", "KOR"]:
    col_l1 = f"Level 1_{lang}"
    col_l2 = f"Level 2_{lang}"
    col_l3 = f"Level 3_{lang}"

    if all(c in df.columns for c in [col_l1, col_l2, col_l3]):
        for _, row in df.iterrows():
            if pd.notna(row[col_l1]) and pd.notna(row[col_l2]) and pd.notna(row[col_l3]):
                label = f"{row[col_l1]} > {row[col_l2]} > {row[col_l3]}"
                all_categories.append({
                    "lang": lang,
                    "label": label,
                    "Level 1": row[col_l1],
                    "Level 2": row[col_l2],
                    "Level 3": row[col_l3]
                })

print(f"총 카테고리 수: {len(all_categories)}")

# 4. 임베딩 생성
texts = [cat["label"] for cat in all_categories]
embeddings = model.encode(texts, show_progress_bar=True)

# 5. FAISS 인덱스 생성
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# 6. 인덱스 저장
faiss.write_index(index, "models/faiss_index.bin")
print("FAISS 인덱스 저장 완료")

# 7. 메타데이터 저장
meta_df = pd.DataFrame(all_categories)
meta_df.to_csv("models/faiss_metadata.csv", index=False)
print("메타데이터 CSV 저장 완료")

# 8. JSON 저장 (딕셔너리 형태)
id_to_category = {
    str(i): {
        "Level 1": cat["Level 1"],
        "Level 2": cat["Level 2"],
        "Level 3": cat["Level 3"],
        "label": cat["label"]
    }
    for i, cat in enumerate(all_categories)
}

with open("models/faiss_mapping.json", "w", encoding="utf-8") as f:
    json.dump(id_to_category, f, ensure_ascii=False, indent=2)
print("faiss_mapping.json 저장 완료")
