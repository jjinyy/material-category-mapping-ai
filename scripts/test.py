import pandas as pd
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

# 경로 설정
MODEL_PATH = "trained_model/"
FAISS_INDEX_PATH = "faiss_index.bin"
FAISS_MAPPING_PATH = "faiss_mapping.json"

# 모델 및 인덱스 로드
model = SentenceTransformer(MODEL_PATH)
index = faiss.read_index(FAISS_INDEX_PATH)

with open(FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
    id_to_category = json.load(f)

# 텍스트 클렌징
def cleanse_text(text):
    return ''.join([c for c in str(text) if c.isalnum() or c.isspace()]).lower().strip()

# 자재명 분류 함수
def classify_material(material_name, language="ENG", top_n=5):
    # 자재명 정제 및 임베딩
    query = cleanse_text(material_name)
    query_embedding = model.encode(query).astype('float32').reshape(1, -1)

    # FAISS 검색
    distances, indices = index.search(query_embedding, top_n)

    # 결과 정리
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if str(idx) in id_to_category:
            category = id_to_category[str(idx)]
            score = 1 / (1 + dist)  # 거리 → 유사도 변환
            results.append({
                "Level 1": category["Level 1"],
                "Level 2": category["Level 2"],
                "Level 3": category["Level 3"],
                "Score": score
            })

    return results
