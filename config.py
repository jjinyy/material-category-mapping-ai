import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models"
TRAINED_MODEL_PATH = MODEL_DIR / "trained_model"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

FAISS_INDEX_PREFIX = MODEL_DIR
DATA_DIR = PROJECT_ROOT / "data"
CATEGORY_CSV = DATA_DIR / "category.csv"
MATERIALS_CSV = DATA_DIR / "materials.csv"
RULE_BASE_JSON = DATA_DIR / "rule_base.json"
USER_FEEDBACK_CSV = DATA_DIR / "user_feedback.csv"
USERS_CSV = DATA_DIR / "users.csv"
TRIPLET_TRAINING_DATA_CSV = DATA_DIR / "triplet_training_data.csv"
NEW_CATEGORY_SUGGESTIONS_CSV = DATA_DIR / "new_category_suggestions.csv"

CLASSIFICATION_TOP_N = 50
RULE_FLOOR = 0.85
RULE_CAP = 0.97
RULE_APPLY_ONLY_IF_IN_ML = True
RULE_OVERRIDE_ML_ORDER = False
RULE_FALLBACK_MARGIN = 0.01

FEEDBACK_WEIGHT = 20
VERIFIED_WEIGHT = 10

EPOCH_CONFIG = {
    0: 5,      # 피드백 0개 (1 → 5로 증가)
    10: 8,     # 피드백 < 10개 (5 → 8로 증가)
    50: 5,     # 피드백 < 50개 (3 → 5로 증가)
    "default": 3  # 피드백 >= 50개 (2 → 3로 증가)
}

LEARNING_RATE = 2e-5
LEARNING_RATE_SCHEDULER = "WarmupLinear"
BATCH_SIZE = 32
BATCH_SIZE_GPU = 64
WARMUP_RATIO = 0.1

def get_faiss_paths(lang: str):
    lang = lang.upper()
    index_path = FAISS_INDEX_PREFIX / f"faiss_{lang}_index.bin"
    mapping_path = FAISS_INDEX_PREFIX / f"faiss_{lang}_mapping.json"
    return str(index_path), str(mapping_path)


def ensure_dirs():
    dirs = [MODEL_DIR, DATA_DIR]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)



