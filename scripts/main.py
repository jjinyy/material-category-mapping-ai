# classification entry

import faiss
import json
import re
from sentence_transformers import SentenceTransformer
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.text_utils import detect_language, preprocess_material_name, cleanse_text

MODEL = None
FAISS_CACHE = {}
RULE_DATA = None
RULE_MTIME = None


def load_model():
    global MODEL

    if MODEL is None:
        model_path = str(config.TRAINED_MODEL_PATH)
        try:
            MODEL = SentenceTransformer(model_path)
            print(f"[MODEL] Load trained model: {model_path}")
        except (FileNotFoundError, OSError) as e:
            print(f"[MODEL] WARN: 학습된 모델을 찾을 수 없습니다 ({e})")
            print(f"[MODEL] 기본 모델을 사용합니다: {config.DEFAULT_MODEL_NAME}")
            try:
                MODEL = SentenceTransformer(config.DEFAULT_MODEL_NAME)
            except Exception as e2:
                raise RuntimeError(f"[MODEL] ERROR: 기본 모델 로드도 실패했습니다: {e2}")
        except Exception as e:
            print(f"[MODEL] ERROR: 모델 로드 중 예상치 못한 오류 발생: {e}")
            raise

    return MODEL


def load_faiss_index(lang):
    global FAISS_CACHE

    if lang in FAISS_CACHE:
        return FAISS_CACHE[lang]

    index_path, mapping_path = config.get_faiss_paths(lang)

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"[FAISS] ERROR: 인덱스 파일을 찾을 수 없습니다: {index_path}\n"
            f"FAISS 인덱스를 먼저 생성해주세요: python train/build_faiss_index.py --lang {lang}"
        )

    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"[FAISS] ERROR: 매핑 파일을 찾을 수 없습니다: {mapping_path}\n"
            f"FAISS 인덱스를 먼저 생성해주세요: python train/build_faiss_index.py --lang {lang}"
        )

    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        raise RuntimeError(f"[FAISS] ERROR: 인덱스 파일 읽기 실패 ({index_path}): {e}")

    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"[FAISS] ERROR: 매핑 파일 JSON 파싱 실패 ({mapping_path}): {e}")
    except Exception as e:
        raise RuntimeError(f"[FAISS] ERROR: 매핑 파일 읽기 실패 ({mapping_path}): {e}")

    FAISS_CACHE[lang] = (index, mapping)
    return index, mapping


def load_rules():
    global RULE_DATA
    global RULE_MTIME

    rule_path = str(config.RULE_BASE_JSON)
    try:
        mtime = os.path.getmtime(rule_path)
    except Exception:
        mtime = None

    # RULE_DATA가 없거나, 파일이 변경되었으면 재로딩
    if RULE_DATA is None or (mtime is not None and RULE_MTIME != mtime):
        try:
            with open(rule_path, "r", encoding="utf-8") as f:
                RULE_DATA = json.load(f)
            RULE_MTIME = mtime
            print(f"[RULE] Loaded: {rule_path} ({len(RULE_DATA)} rules)")
        except FileNotFoundError:
            print(f"[RULE] WARN: 규칙 파일을 찾을 수 없습니다: {rule_path}")
            print("[RULE] 규칙 기반 매칭을 사용하지 않습니다.")
            RULE_DATA = []
            RULE_MTIME = mtime
        except json.JSONDecodeError as e:
            print(f"[RULE] ERROR: 규칙 파일 JSON 파싱 실패 ({rule_path}): {e}")
            print("[RULE] 규칙 기반 매칭을 사용하지 않습니다.")
            RULE_DATA = []
            RULE_MTIME = mtime
        except Exception as e:
            print(f"[RULE] ERROR: 규칙 파일 읽기 실패 ({rule_path}): {e}")
            print("[RULE] 규칙 기반 매칭을 사용하지 않습니다.")
            RULE_DATA = []
            RULE_MTIME = mtime

    return RULE_DATA


def apply_rule(material_name, material_type, ml_results, lang):
    """Rule match → that category 1st, score = ML top + margin."""
    RULE_CAP = getattr(config, "RULE_CAP", 0.97)
    only_if_in_ml = getattr(config, "RULE_APPLY_ONLY_IF_IN_ML", True)
    fallback_margin = getattr(config, "RULE_FALLBACK_MARGIN", 0.01)

    rules = load_rules()
    name = material_name.lower()

    matched_code = None
    for rule in rules:
        if rule.get("type") and material_type and rule["type"] != material_type:
            continue

        for kw in rule.get("keywords", []):
            kw_lower = kw.lower()
            if any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in kw):
                if kw in name or kw_lower in name:
                    matched_code = rule["code"]
                    break
            else:
                words = re.findall(r"\w+", name)
                if kw_lower in words or kw_lower in name:
                    matched_code = rule["code"]
                    break

        if matched_code:
            break

    if not matched_code:
        return ml_results

    existing_score = None
    for r in ml_results:
        if r["CODE"] == matched_code:
            existing_score = r["Score"]
            break

    _, mapping = FAISS_CACHE[lang]
    matched_item = None
    for item in mapping.values():
        if item["CODE"] == matched_code:
            matched_item = item
            break

    def _make_rule_row(score, source="rule"):
        return {
            "CODE": matched_code,
            "TYPE": material_type,
            "L1": matched_item.get("L1", "") if matched_item else "",
            "L2": matched_item.get("L2", "") if matched_item else "",
            "L3": matched_item.get("L3", "") if matched_item else "",
            "L4": matched_item.get("L4", "") if matched_item else "",
            "Score": score,
            "score_source": source,
        }

    if only_if_in_ml and existing_score is None:
        top_ml_score = max((r["Score"] for r in ml_results), default=0.0)
        score = min(top_ml_score + fallback_margin, RULE_CAP)
        fallback_row = _make_rule_row(score, "rule_fallback")
        out = [r for r in ml_results if r["CODE"] != matched_code]
        out.append(fallback_row)
        out.sort(key=lambda x: x["Score"], reverse=True)
        return out

    top_ml_score = max((r["Score"] for r in ml_results), default=0.0)
    score = min(top_ml_score + fallback_margin, RULE_CAP)
    override = _make_rule_row(score, "rule_override")
    filtered = [r for r in ml_results if r["CODE"] != matched_code]
    return [override] + filtered




def classify_material(material_name, selected_type=None, top_n=None):
    if not material_name or not isinstance(material_name, str):
        raise ValueError("[ERROR] 자재명은 비어있지 않은 문자열이어야 합니다.")
    
    if top_n is None:
        top_n = config.CLASSIFICATION_TOP_N
    
    try:
        top_n = int(top_n)
        if top_n <= 0:
            raise ValueError(f"[ERROR] top_n은 양수여야 합니다: {top_n}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"[ERROR] top_n은 정수여야 합니다: {e}")

    try:
        processed = cleanse_text(preprocess_material_name(material_name))
        if not processed:
            raise ValueError("[ERROR] 전처리 후 자재명이 비어있습니다.")
        
        lang = detect_language(processed)
        print(f"[LANG] 감지: {lang}")

        model = load_model()
        index, mapping = load_faiss_index(lang)

        emb = model.encode(processed, convert_to_numpy=True).astype("float32").reshape(1, -1)
        distances, indices = index.search(emb, top_n)

        results = []

        for dist, idx in zip(distances[0], indices[0]):
            cat = mapping.get(str(idx))
            if not cat:
                continue

            if selected_type and cat.get("TYPE") != selected_type:
                continue

            results.append({
                "CODE": cat["CODE"],
                "TYPE": cat["TYPE"],
                "L1": cat.get("L1", ""),
                "L2": cat.get("L2", ""),
                "L3": cat.get("L3", ""),
                "L4": cat.get("L4", ""),
                "Score": 1 / (1 + dist)
            })

        try:
            results = apply_rule(material_name, selected_type, results, lang)
        except Exception as e:
            print(f"[RULE] WARN: 규칙 적용 중 오류 발생 (계속 진행): {e}")

        results = sorted(results, key=lambda x: x["Score"], reverse=True)
        if not results:
            print(f"[WARN] 분류 결과가 없습니다. 입력: '{material_name}', 타입: {selected_type}")
            print(f"[WARN] 전처리 결과: '{processed}', 언어: {lang}")

        return results

    except (FileNotFoundError, RuntimeError) as e:
        # 모델/인덱스 로드 실패는 상위로 전파
        raise
    except Exception as e:
        raise RuntimeError(f"[ERROR] 분류 중 예상치 못한 오류 발생: {e}")


if __name__ == "__main__":
    q = input("자재명을 입력하세요: ")
    for r in classify_material(q):
        print(r)

