"""
main.py
-------

학습된 문장 임베딩 모델과 구축된 FAISS 인덱스를 로드하여
사용자로부터 입력받은 자재명을 기반으로 관련 카테고리 분류

1. 학습된 문장 임베딩 모델 로드 (models/trained_model)
2. FAISS 인덱스 로드 (models/faiss_index.bin)
3. 인덱스 ID ➔ 카테고리 매핑 정보 로드 (models/faiss_mapping.json)
4. 자재명 텍스트 클렌징 함수 정의
5. 입력 자재명에 대한 Top-N 카테고리 검색 및 결과 반환 함수 정의 (classify_material)

"""

import faiss
import json
from sentence_transformers import SentenceTransformer
# from huggingface_hub import hf_hub_download
# import os

# 경로 설정
# 수정
# token = os.getenv("HF_TOKEN")
token = "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"

# FAISS 인덱스 파일 경로
FAISS_INDEX_PATH = "models/faiss_index.bin"
FAISS_MAPPING_PATH = "models/faiss_mapping.json"

# Hugging Face에서 바로 모델 로드
# 1. 모델 및 인덱스 로드
try:
    model = SentenceTransformer(
        "jjinny/categoryMapping",
        use_auth_token=token
    )
    index = faiss.read_index("models/faiss_index.bin")
except Exception as e:
    print(f"모델 또는 인덱스 로드 실패: {e}")
    exit(1)

# 2. 카테고리 매핑 로드
with open(FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
    id_to_category = json.load(f)

# 3. 텍스트 클렌징
def cleanse_text(text):
    """
    입력된 텍스트를 소문자화 및 알파벳/숫자/공백 이외의 문자를 제거하여 클렌징.
    """
    return ''.join([c for c in str(text) if c.isalnum() or c.isspace()]).lower().strip()

# 4. 자재명 분류 함수
def classify_material(material_name, language="ENG", top_n=5):
    """
    자재명을 입력받아 가장 관련성 높은 Top-N 카테고리 반환

    Args:
        material_name (str): 분류하고자 하는 자재명
        language (str): 검색 언어 설정 (현재 미사용, 확장 대비)
        top_n (int): 반환할 상위 결과 수

    Returns:
        list[dict]: 카테고리 및 유사도 점수가 포함된 결과 리스트
    """
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
            score = 1 / (1 + dist)  # 거리(0에 가까울수록 유사함)를 유사도 점수로 변환
            results.append({
                "Level 1": category["Level 1"],
                "Level 2": category["Level 2"],
                "Level 3": category["Level 3"],
                "Score": score
            })

    return results

if __name__ == "__main__":
    # 테스트용 자재명
    test_material = "banana puree"
    print(f"\n입력 자재명: {test_material}\n")
    # 분류 결과 출력
    results = classify_material(test_material, language="ENG")
    for res in results:
        print(f"Level 1: {res['Level 1']}, Level 2: {res['Level 2']}, Level 3: {res['Level 3']}, Score: {res['Score']:.3f}")