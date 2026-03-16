"""
프로젝트 설정 파일
하드코딩된 값들을 중앙에서 관리.
"""

import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent

# ============================================
# 파일 경로 설정
# ============================================
# 모델 관련
MODEL_DIR = PROJECT_ROOT / "models"
TRAINED_MODEL_PATH = MODEL_DIR / "trained_model"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# FAISS 인덱스
# 실제 파일은 models/faiss_KR_index.bin 형태로 저장됨
FAISS_INDEX_PREFIX = MODEL_DIR

# 데이터 파일
DATA_DIR = PROJECT_ROOT / "data"
CATEGORY_CSV = DATA_DIR / "category.csv"
MATERIALS_CSV = DATA_DIR / "materials.csv"
RULE_BASE_JSON = DATA_DIR / "rule_base.json"
USER_FEEDBACK_CSV = DATA_DIR / "user_feedback.csv"
USERS_CSV = DATA_DIR / "users.csv"  # 로그인 계정 (username, password_hash)
TRIPLET_TRAINING_DATA_CSV = DATA_DIR / "triplet_training_data.csv"
NEW_CATEGORY_SUGGESTIONS_CSV = DATA_DIR / "new_category_suggestions.csv"

# ============================================
# 모델 설정
# ============================================
# 분류 설정
CLASSIFICATION_TOP_N = 50  # 기본 반환 결과 수

# ============================================
# 규칙 기반 매칭 설정 (ML 우선, 규칙은 지지선)
# ============================================
RULE_FLOOR = 0.85   # 규칙으로 1등 올릴 때 최소 표시 점수 (규칙 코드가 ML에 있을 때만)
RULE_CAP = 0.97     # 규칙 적용 시 최대 점수 (절대 1.0 금지)
# 규칙 코드가 ML 결과에 없으면 1등으로 올리지 않고, 지지선 점수로만 목록에 추가
RULE_APPLY_ONLY_IF_IN_ML = True
# False = ML 우선. ML 1등이 규칙과 다르면 규칙으로 순서 변경 안 함
RULE_OVERRIDE_ML_ORDER = False
# ML 상위에 없을 때 규칙 지지선 점수 = (ML 1등 점수 + RULE_FALLBACK_MARGIN), 상한 RULE_CAP
# (바닥 없이 항상 "ML 1등보다 살짝만 높은" 점수로 표시)
RULE_FALLBACK_MARGIN = 0.01  # 0.01 ≒ 화면 1% (예: ML 1등 6.43% → 규칙 7.43%)

# ============================================
# 학습 데이터 생성 설정
# ============================================
FEEDBACK_WEIGHT = 20  # 피드백 기반 Triplet 가중치 (반복 횟수)
VERIFIED_WEIGHT = 10  # 검증된 매핑 데이터(materials.csv) 가중치 (반복 횟수)

# ============================================
# 학습 설정
# ============================================
# Epoch 자동 조정 기준 (피드백 수에 따라)
# 정확도 향상을 위해 기본 epoch 수를 증가시킴
EPOCH_CONFIG = {
    0: 5,      # 피드백 0개 (1 → 5로 증가)
    10: 8,     # 피드백 < 10개 (5 → 8로 증가)
    50: 5,     # 피드백 < 50개 (3 → 5로 증가)
    "default": 3  # 피드백 >= 50개 (2 → 3로 증가)
}

# 학습률 설정
LEARNING_RATE = 2e-5  # 기본 학습률 (너무 높으면 불안정, 너무 낮으면 학습 느림)
LEARNING_RATE_SCHEDULER = "WarmupLinear"  # 학습률 스케줄러

# 배치 사이즈 설정
BATCH_SIZE = 32  # 기본 배치 사이즈 (CPU 사용 시)
BATCH_SIZE_GPU = 64  # GPU 사용 시 배치 사이즈 (GPU 메모리에 따라 조정 가능)

# Warmup 비율
WARMUP_RATIO = 0.1  # 전체 스텝의 10%를 warmup으로 사용

# ============================================
# 유틸리티 함수
# ============================================
def get_faiss_paths(lang: str):
    """
    언어별 FAISS 인덱스 파일 경로를 반환합니다.
    
    Args:
        lang: 언어 코드 ("KR" 또는 "EN")
    
    Returns:
        tuple: (index_path, mapping_path)
    """
    lang = lang.upper()
    index_path = FAISS_INDEX_PREFIX / f"faiss_{lang}_index.bin"
    mapping_path = FAISS_INDEX_PREFIX / f"faiss_{lang}_mapping.json"
    return str(index_path), str(mapping_path)


def ensure_dirs():
    """필수 디렉토리가 존재하는지 확인하고 없으면 생성합니다."""
    dirs = [MODEL_DIR, DATA_DIR]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)



