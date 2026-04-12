"""
admin/retrain.py
----------------
전체 파이프라인 자동 실행 스크립트 (KR/EN 인덱스 모두 생성)

1) Triplet 데이터 생성
2) Triplet 기반 모델 학습
3) FAISS 인덱스 생성 (KR)
4) FAISS 인덱스 생성 (EN)

subprocess 대신 직접 함수를 호출하여 실행합니다.
인코딩 문제 없이 파이참에서도 효율적으로 실행 가능합니다.
"""

import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate.generate_triplet_data import generate
from train.train_triplet_model import train_triplet_model
from train.build_faiss_index import build_faiss_index
import config


def run_step(description, func, *args, **kwargs):
    """
    각 단계를 실행하고 에러를 처리합니다.
    
    Args:
        description: 단계 설명
        func: 실행할 함수
        *args, **kwargs: 함수에 전달할 인자
    """
    print(f"\n▶ {description}")
    
    try:
        func(*args, **kwargs)
        print(f"완료: {description}")
    except Exception as e:
        print(f"실패: {description}")
        print(f"   에러: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    print("\n===================================")
    print("   Material Category 재학습 시작")
    print("===================================\n")

    # 1) Triplet 생성
    run_step(
        "Triplet 데이터 생성 (basic + hard)",
        generate,
        materials=str(config.MATERIALS_CSV),
        categories=str(config.CATEGORY_CSV),
        output=str(config.TRIPLET_TRAINING_DATA_CSV),
        mode="all",
        include_feedback=True
    )

    # 2) 모델 학습
    run_step(
        "Triplet 모델 학습 시작",
        train_triplet_model,
        triplet_path=str(config.TRIPLET_TRAINING_DATA_CSV),
        trained_model_path=str(config.TRAINED_MODEL_PATH),
        epochs=None,  # config.EPOCH_CONFIG에 따라 자동 결정
        batch_size=None,  # config.BATCH_SIZE 사용
        learning_rate=None  # config.LEARNING_RATE 사용
    )

    # 3) KR 인덱스 생성
    run_step(
        "FAISS 인덱스 생성 (KR)",
        build_faiss_index,
        lang="KR",
        model_path=str(config.TRAINED_MODEL_PATH),
        category_path=str(config.CATEGORY_CSV),
        output_prefix=str(config.FAISS_INDEX_PREFIX)
    )

    # 4) EN 인덱스 생성
    run_step(
        "FAISS 인덱스 생성 (EN)",
        build_faiss_index,
        lang="EN",
        model_path=str(config.TRAINED_MODEL_PATH),
        category_path=str(config.CATEGORY_CSV),
        output_prefix=str(config.FAISS_INDEX_PREFIX)
    )

    print("\n===================================")
    print("      전체 파이프라인 완료!")
    print("===================================\n")


if __name__ == "__main__":
    main()
