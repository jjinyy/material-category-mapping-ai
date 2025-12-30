import pandas as pd
import numpy as np
import faiss
import os
import sys
import json
import re
import argparse
from sentence_transformers import SentenceTransformer

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.text_utils import cleanse_text


def build_faiss_index(
    lang="KR",
    model_path=None,
    category_path=None,
    output_prefix=None
):
    """
    FAISS 인덱스를 생성합니다.
    
    Args:
        lang: 언어 선택 ("KR" 또는 "EN", 기본값: "KR")
        model_path: 학습된 모델 경로 (기본값: config.TRAINED_MODEL_PATH)
        category_path: 카테고리 파일 경로 (기본값: config.CATEGORY_CSV)
        output_prefix: 저장 prefix (기본값: config.FAISS_INDEX_PREFIX)
    """
    if model_path is None:
        model_path = str(config.TRAINED_MODEL_PATH)
    if category_path is None:
        category_path = str(config.CATEGORY_CSV)
    if output_prefix is None:
        output_prefix = config.FAISS_INDEX_PREFIX
    
    lang = lang.upper()
    # Path 객체를 문자열로 변환하여 경로 생성
    if isinstance(output_prefix, (str, os.PathLike)):
        output_prefix = str(output_prefix)
    save_prefix = os.path.join(output_prefix, f"faiss_{lang}") if output_prefix else f"faiss_{lang}"

    # --------------------
    # 모델 로드
    # --------------------
    try:
        print(f"[MODEL] 모델 로드: {model_path}")
        model = SentenceTransformer(model_path)
    except (FileNotFoundError, OSError) as e:
        print(f"[MODEL] WARN: 학습된 모델을 찾을 수 없습니다 ({e})")
        print(f"[MODEL] 기본 모델을 사용합니다: {config.DEFAULT_MODEL_NAME}")
        try:
            model = SentenceTransformer(config.DEFAULT_MODEL_NAME)
        except Exception as e2:
            print(f"[MODEL] ERROR: 기본 모델 로드도 실패했습니다: {e2}")
            raise RuntimeError(f"모델을 로드할 수 없습니다: {e2}")
    except Exception as e:
        print(f"[MODEL] ERROR: 모델 로드 중 예상치 못한 오류: {e}")
        raise

    # --------------------
    # category.csv 로드
    # --------------------
    if not os.path.exists(category_path):
        raise FileNotFoundError(f"[ERROR] 카테고리 파일을 찾을 수 없습니다: {category_path}")

    try:
        df = pd.read_csv(category_path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise ValueError(f"[ERROR] 카테고리 파일이 비어있습니다: {category_path}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] 카테고리 파일 읽기 실패 ({category_path}): {e}")

    if df.empty:
        raise ValueError(f"[ERROR] 카테고리 데이터가 없습니다: {category_path}")

    if lang == "KR":
        L1 = "L1_KR"
        L2 = "L2_KR"
        L3 = "L3_KR"
        L4 = "L4_KR"
    else:
        L1 = "L1_EN"
        L2 = "L2_EN"
        L3 = "L3_EN"
        L4 = "L4_EN"

    # --------------------
    # 카테고리 라벨 생성
    # --------------------
    categories = []

    for _, row in df.iterrows():
        parts = [
            str(row[L1]) if pd.notna(row[L1]) else "",
            str(row[L2]) if pd.notna(row[L2]) else "",
            str(row[L3]) if pd.notna(row[L3]) else "",
        ]
        if L4 in row and pd.notna(row[L4]):
            parts.append(str(row[L4]))

        label_text = cleanse_text(" ".join(parts))

        categories.append({
            "CODE": row["CODE"],
            "TYPE": row["TYPE"],
            "L1": row[L1],
            "L2": row[L2],
            "L3": row[L3],
            "L4": row.get(L4, ""),
            "label": label_text
        })

    # --------------------
    # 임베딩 생성
    # --------------------
    if not categories:
        raise ValueError(f"[ERROR] 생성된 카테고리가 없습니다. 카테고리 파일을 확인해주세요: {category_path}")

    try:
        texts = [c["label"] for c in categories]
        print(f"[EMBEDDING] {len(texts)}개 카테고리 임베딩 생성 중...")
        emb = model.encode(texts, convert_to_numpy=True).astype("float32")
        print(f"[EMBEDDING] 완료: {emb.shape}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] 임베딩 생성 실패: {e}")

    try:
        index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb)
        print(f"[FAISS] 인덱스 생성 완료: {index.ntotal}개 벡터")
    except Exception as e:
        raise RuntimeError(f"[ERROR] FAISS 인덱스 생성 실패: {e}")

    # --------------------
    # 저장
    # --------------------
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

    try:
        index_path = f"{save_prefix}_index.bin"
        faiss.write_index(index, index_path)
        print(f"[SAVE] 인덱스 저장: {index_path}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] 인덱스 파일 저장 실패 ({index_path}): {e}")

    try:
        mapping_path = f"{save_prefix}_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump({str(i): categories[i] for i in range(len(categories))},
                      f,
                      ensure_ascii=False,
                      indent=2)
        print(f"[SAVE] 매핑 저장: {mapping_path}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] 매핑 파일 저장 실패 ({mapping_path}): {e}")

    print(f"[DONE] {lang} 언어 인덱스 생성 완료!")


# ============================================
# Entry point (명령줄 실행용)
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(config.TRAINED_MODEL_PATH),
                        help="학습된 모델 경로")
    parser.add_argument("--category", type=str, default=str(config.CATEGORY_CSV),
                        help="카테고리 파일 경로")
    parser.add_argument("--lang", type=str, default="KR", help="KR 또는 EN 선택")
    parser.add_argument("--output_prefix", type=str, default=str(config.FAISS_INDEX_PREFIX),
                        help="저장 prefix")
    args = parser.parse_args()

    build_faiss_index(
        lang=args.lang,
        model_path=args.model,
        category_path=args.category,
        output_prefix=args.output_prefix
    )
