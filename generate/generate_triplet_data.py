import pandas as pd
import random
import argparse
import os
import sys
import re
from itertools import combinations
from datetime import datetime

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.text_utils import detect_language, clean_name

# ======================================================
# Category seed anchors (개념 공간 형성용)
# ======================================================
CATEGORY_SEED_ANCHORS = {
    "나물류": ["고사리", "취나물", "도라지", "시래기", "곤드레", "산나물", "나물"],
    "버섯류": ["표고버섯", "느타리버섯", "팽이버섯", "새송이버섯"],
}

# ======================================================
# 카테고리 라벨 생성 (언어 고정)
# ======================================================
def build_label(row, target_lang):
    prefix = "KR" if target_lang == "kor" else "EN"
    parts = []
    for i in range(1, 5):
        col = f"L{i}_{prefix}"
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " > ".join(parts)

# ======================================================
# Triplet builder (anchor 기준 언어 고정)
# ======================================================
def build_triplet(a, p, n):
    lang = a["lang"]

    anchor_text = a["clean_name"]
    positive = build_label(p, lang)
    negative = build_label(n, lang)

    if not anchor_text or not positive or not negative:
        return None

    return {
        "anchor": anchor_text,
        "positive": positive,
        "negative": negative,
        "TYPE": a["TYPE"],
        "code": a["code"],
        "source": a.get("source", "material"),
    }

# ======================================================
# Basic Triplet
# ======================================================
def generate_basic(df):
    triples = []
    for (_, _, lang), group in df.groupby(["TYPE", "code", "lang"]):
        if len(group) < 2:
            continue

        group = group.sample(min(len(group), 5), random_state=42)

        for _, a in group.iterrows():
            pos_pool = group[group["clean_name"] != a["clean_name"]]
            if pos_pool.empty:
                continue

            p = pos_pool.sample(1).iloc[0]

            neg_pool = df[
                (df["TYPE"] == a["TYPE"]) &
                (df["code"] != a["code"]) &
                (df["lang"] == lang)
            ]
            if neg_pool.empty:
                continue

            n = neg_pool.sample(1).iloc[0]
            t = build_triplet(a, p, n)
            if t:
                triples.append(t)

    return triples

# ======================================================
# Hard Triplet
# ======================================================
def generate_hard(df):
    triples = []
    for (TYPE, code, lang), group in df.groupby(["TYPE", "code", "lang"]):
        if len(group) < 2:
            continue

        names = group["clean_name"].unique()
        if len(names) < 2:
            continue

        pairs = list(combinations(names, 2))[:5]

        prefix = "KR" if lang == "kor" else "EN"
        L1_col = f"L1_{prefix}"
        L2_col = f"L2_{prefix}"

        for a_name, p_name in pairs:
            a = group[group["clean_name"] == a_name].iloc[0]
            p = group[group["clean_name"] == p_name].iloc[0]

            neg_pool = df[
                (df["TYPE"] == TYPE) &
                (df["code"] != code) &
                (df["lang"] == lang) &
                ((df[L1_col] == a[L1_col]) | (df[L2_col] == a[L2_col]))
            ]
            if neg_pool.empty:
                continue

            n = neg_pool.sample(1).iloc[0]
            t = build_triplet(a, p, n)
            if t:
                triples.append(t)

    return triples

# ======================================================
# Verified Triplet (materials.csv의 검증된 매핑 활용)
# ======================================================
def generate_from_verified_materials(df, verified_weight=None):
    """
    materials.csv에 이미 매핑된 데이터를 피드백처럼 가중치를 주어 활용합니다.
    이 데이터는 이미 검증된 매핑이므로 신뢰도가 높습니다.
    
    Args:
        df: materials와 categories가 merge된 데이터프레임
        verified_weight: 가중치 (기본값: config.VERIFIED_WEIGHT)
    
    Returns:
        list: triplet 리스트
    """
    if verified_weight is None:
        verified_weight = config.VERIFIED_WEIGHT
    
    triples = []
    
    # code가 있는 데이터만 사용 (매핑된 데이터)
    df_verified = df[df["code"].notna() & (df["code"] != "")]
    
    print(f"[VERIFIED] 검증된 매핑 데이터: {len(df_verified):,}개")
    
    for _, row in df_verified.iterrows():
        material_name = str(row["clean_name"]).strip()
        material_type = row["TYPE"]
        code = row["code"]
        lang = row["lang"]
        
        if not material_name or not code:
            continue
        
        # 카테고리 라벨 생성
        pos_label = build_label(row, lang)
        if not pos_label:
            continue
        
        # Negative 샘플링 (같은 TYPE, 다른 code)
        neg_pool = df_verified[
            (df_verified["TYPE"] == material_type) &
            (df_verified["code"] != code) &
            (df_verified["lang"] == lang)
        ]
        
        if neg_pool.empty:
            continue
        
        # 가중치만큼 반복 생성
        for _ in range(verified_weight):
            n = neg_pool.sample(1, random_state=42).iloc[0]
            neg_label = build_label(n, lang)
            
            t = {
                "anchor": material_name,
                "positive": pos_label,
                "negative": neg_label,
                "TYPE": material_type,
                "code": code,
                "source": "verified",
            }
            triples.append(t)
    
    return triples


# ======================================================
# Feedback Triplet (NEW only + seed anchor)
# ======================================================
def generate_from_feedback(df_fb, df_all, feedback_weight=None):
    if feedback_weight is None:
        feedback_weight = config.FEEDBACK_WEIGHT
    
    triples = []

    for _, row in df_fb.iterrows():
        material_name = str(row["material_name"]).strip()
        material_type = row["material_type"]
        code = row["code"]

        lang = row.get("lang")
        if not lang:
            lang = detect_language(material_name, return_format="full")

        pos_label = " > ".join(
            [str(row.get(f"L{i}", "")).strip() for i in range(1, 5)]
        ).strip(" > ")
        if not pos_label:
            continue

        # --- anchor 후보 생성 (★ 핵심 수정) ---
        anchor_names = [
            material_name,
            f"{material_name} 나물",
            f"나물 {material_name}",
            f"{material_name} 산나물",
        ]

        # seed anchor는 보조적으로만 사용
        l3 = str(row.get("L3", "")).strip()
        if l3 in CATEGORY_SEED_ANCHORS:
            anchor_names += CATEGORY_SEED_ANCHORS[l3]

        neg_pool = df_all[
            (df_all["TYPE"] == material_type) &
            (df_all["code"] != code) &
            (df_all["lang"] == lang)
        ]
        if neg_pool.empty:
            continue

        for anchor_name in set(anchor_names):
            a = {
                "clean_name": clean_name(anchor_name),
                "lang": lang,
                "TYPE": material_type,
                "code": code,
                "source": "feedback",
            }
            n = neg_pool.sample(1, random_state=42).iloc[0]

            for _ in range(feedback_weight):
                t = {
                    "anchor": a["clean_name"],
                    "positive": pos_label,
                    "negative": build_label(n, lang),
                    "TYPE": material_type,
                    "code": code,
                    "source": "feedback",
                }
                triples.append(t)

    return triples

# ======================================================
# Main generate
# ======================================================
def generate(materials=None, categories=None, output=None, mode="all", include_feedback=True):
    if materials is None:
        materials = str(config.MATERIALS_CSV)
    if categories is None:
        categories = str(config.CATEGORY_CSV)
    if output is None:
        output = str(config.TRIPLET_TRAINING_DATA_CSV)

    df_m = pd.read_csv(materials, encoding="utf-8-sig")
    df_c = pd.read_csv(categories, encoding="utf-8-sig")

    df_m.rename(columns={"CATEGORY_CODE": "code"}, inplace=True)
    df_m["clean_name"] = df_m["MATERIAL_DESC"].apply(clean_name)
    df_m["lang"] = df_m["MATERIAL_DESC"].apply(lambda x: detect_language(x, return_format="full"))

    df = df_m.merge(
        df_c,
        left_on=["code", "TYPE"],
        right_on=["CODE", "TYPE"],
        how="left",
    ).drop(columns=["CODE"])

    df = df.drop_duplicates(subset=["clean_name", "code", "TYPE"])
    df["source"] = "material"

    triplets = []

    # 검증된 매핑 데이터를 피드백처럼 가중치를 주어 적극 활용
    print("\n[Verified Triplets 생성] (검증된 매핑 데이터 활용)")
    verified_triplets = generate_from_verified_materials(df)
    triplets += verified_triplets
    print(f"[VERIFIED] 검증된 매핑 기반 Triplet 생성: {len(verified_triplets):,}개 (가중치: {config.VERIFIED_WEIGHT}x)")

    if mode in ["basic", "all"]:
        print("\n[Basic Triplets 생성]")
        basic_triplets = generate_basic(df)
        triplets += basic_triplets
        print(f"[BASIC] Basic Triplet 생성: {len(basic_triplets):,}개")

    if mode in ["hard", "all"]:
        print("\n[Hard Triplets 생성]")
        hard_triplets = generate_hard(df)
        triplets += hard_triplets
        print(f"[HARD] Hard Triplet 생성: {len(hard_triplets):,}개")

    fb_path = str(config.USER_FEEDBACK_CSV)
    if include_feedback and os.path.exists(fb_path):
        df_fb = pd.read_csv(fb_path, encoding="utf-8-sig")
        new_fb = df_fb[df_fb["status"] == "NEW"]
        print(f"[FEEDBACK] 새로운 피드백: {len(new_fb)}개 (재학습에 반영)")

        if not new_fb.empty:
            feedback_triplets = generate_from_feedback(new_fb, df)
            triplets += feedback_triplets
            print(f"[FEEDBACK] 피드백 기반 Triplet 생성: {len(feedback_triplets)}개 (가중치: {config.FEEDBACK_WEIGHT}x)")
            # 피드백 사용 후 상태 변경 (다음 재학습에서는 제외)
            df_fb.loc[new_fb.index, "status"] = "USED"
            df_fb.loc[new_fb.index, "used_at"] = datetime.now().isoformat()
            df_fb.to_csv(fb_path, index=False, encoding="utf-8-sig")
            print(f"[FEEDBACK] 피드백 상태 업데이트: NEW → USED ({len(new_fb)}개)")

    out_df = pd.DataFrame(triplets).drop_duplicates()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    out_df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"\n완료! Triplet 총 {len(out_df)}개 생성")
    print(f"저장 위치: {output}")

# ======================================================
# Entry
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all")
    args = parser.parse_args()

    generate(
        materials=str(config.MATERIALS_CSV),
        categories=str(config.CATEGORY_CSV),
        output=str(config.TRIPLET_TRAINING_DATA_CSV),
        mode=args.mode,
        include_feedback=True,
    )
