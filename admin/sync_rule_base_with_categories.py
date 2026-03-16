import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _norm(s: str) -> str:
    return str(s or "").strip().lower()


def _load_categories(category_csv: str) -> pd.DataFrame:
    df = pd.read_csv(category_csv, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"category.csv is empty: {category_csv}")
    required = {"TYPE", "CODE", "L1_KR", "L1_EN", "L2_KR", "L2_EN", "L3_KR", "L3_EN"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"category.csv missing columns: {sorted(list(missing))}")
    return df


def _build_label_maps(df_cat: pd.DataFrame) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, str]]:
    """
    Returns:
      label_to_codes: normalized label -> list of (code, type)
      code_to_type: code -> type
    """
    code_to_type = dict(zip(df_cat["CODE"].astype(str), df_cat["TYPE"].astype(str)))

    label_to_codes: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def add_label(label: Any, code: str, rtype: str):
        n = _norm(label)
        if not n:
            return
        label_to_codes[n].append((code, rtype))

    for _, row in df_cat.iterrows():
        code = str(row["CODE"])
        rtype = str(row["TYPE"])
        for col in ["L3_KR", "L3_EN", "L4_KR", "L4_EN"]:
            if col in row:
                add_label(row.get(col), code, rtype)

    return label_to_codes, code_to_type


def sync_rule_base_with_categories(
    category_csv: str,
    rule_base_json: str,
    output_json: str | None = None,
    report_json: str | None = None,
) -> dict:
    """
    category.csv를 기준으로 rule_base.json을 정리/동기화합니다.

    수행 작업:
    1) category.csv에 없는 code를 참조하는 규칙은 keyword(L3/L4)로 code 자동 교정 시도
    2) keyword 충돌(같은 keyword가 여러 code)에 대해, category.csv(L3/L4) 기준으로 "정답 code"가 있으면
       그 code 규칙에만 keyword를 남기고 나머지 규칙에서는 제거
    3) 그래도 남는 충돌은 파일 순서 기준(첫 등장 rule 유지, 이후 rule에서 keyword 제거)

    Note:
    - `scripts/main.py`는 rule keywords 매칭만 사용하므로, 여기서는 keywords 중심으로 정리합니다.
    """
    df_cat = _load_categories(category_csv)
    label_to_codes, code_to_type = _build_label_maps(df_cat)
    codes = set(code_to_type.keys())

    rules: List[Dict[str, Any]] = json.loads(Path(rule_base_json).read_text(encoding="utf-8"))
    if not isinstance(rules, list):
        raise ValueError("rule_base.json must be a JSON array (list)")

    # 1) missing code auto-fix
    missing_code_before = []
    missing_code_fixed = []
    missing_code_unfixed = []

    for idx, r in enumerate(rules):
        if not isinstance(r, dict):
            continue
        code = str(r.get("code", "")).strip()
        rtype = str(r.get("type", "")).strip()
        kws = r.get("keywords", [])
        if not code or not rtype or not isinstance(kws, list):
            continue

        if code in codes:
            continue

        missing_code_before.append({"idx": idx, "code": code, "type": rtype, "keywords": kws[:10]})

        # Try infer from keywords that are exact category labels (L3/L4)
        candidates: List[Tuple[str, str]] = []
        for kw in kws:
            for cand_code, cand_type in label_to_codes.get(_norm(kw), []):
                if cand_type == rtype:
                    candidates.append((cand_code, cand_type))

        cand_codes = sorted(list({c[0] for c in candidates}))
        if len(cand_codes) == 1:
            new_code = cand_codes[0]
            r["code"] = new_code
            missing_code_fixed.append({"idx": idx, "old_code": code, "new_code": new_code})
        else:
            missing_code_unfixed.append(
                {"idx": idx, "code": code, "type": rtype, "candidates": cand_codes[:20], "keywords": kws[:10]}
            )

    # 2) keyword conflicts resolution
    # Build kw -> list of (rule_idx, code, type)
    kw_occurrences: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for idx, r in enumerate(rules):
        if not isinstance(r, dict):
            continue
        code = str(r.get("code", "")).strip()
        rtype = str(r.get("type", "")).strip()
        kws = r.get("keywords", [])
        if not code or not rtype or not isinstance(kws, list):
            continue
        for kw in kws:
            n = _norm(kw)
            if n:
                kw_occurrences[n].append((idx, code, rtype))

    keyword_conflicts_before = [
        {"keyword": kw, "occurrences": occs}
        for kw, occs in kw_occurrences.items()
        if len({o[1] for o in occs}) > 1
    ]

    removed_keywords = []  # entries {keyword, removed_from_idx, kept_idx, reason}

    for kw, occs in kw_occurrences.items():
        codes_involved = sorted(list({o[1] for o in occs}))
        if len(codes_involved) <= 1:
            continue

        # If keyword maps uniquely to a code in categories (same type), keep only there
        cat_hits = label_to_codes.get(kw, [])
        cat_codes = sorted(list({c for c, _t in cat_hits}))

        keep_rule_idx: int | None = None
        if len(cat_codes) == 1:
            target_code = cat_codes[0]
            for ridx, rcode, _rtype in occs:
                if rcode == target_code:
                    keep_rule_idx = ridx
                    break

        if keep_rule_idx is None:
            # default: keep first occurrence in file order
            keep_rule_idx = min([o[0] for o in occs])

        # Remove kw from all other rules
        for ridx, rcode, _rtype in occs:
            if ridx == keep_rule_idx:
                continue
            r = rules[ridx]
            kws = r.get("keywords", [])
            if isinstance(kws, list):
                new_kws = [k for k in kws if _norm(k) != kw]
                if len(new_kws) != len(kws):
                    r["keywords"] = new_kws
                    removed_keywords.append(
                        {
                            "keyword": kw,
                            "removed_from_idx": ridx,
                            "removed_from_code": rcode,
                            "kept_idx": keep_rule_idx,
                            "reason": "category_unique_match" if len(cat_codes) == 1 else "first_rule_wins",
                        }
                    )

    # Recompute conflicts after removal
    kw_occ_after: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for idx, r in enumerate(rules):
        if not isinstance(r, dict):
            continue
        code = str(r.get("code", "")).strip()
        rtype = str(r.get("type", "")).strip()
        kws = r.get("keywords", [])
        if not code or not rtype or not isinstance(kws, list):
            continue
        for kw in kws:
            n = _norm(kw)
            if n:
                kw_occ_after[n].append((idx, code, rtype))

    keyword_conflicts_after = [
        {"keyword": kw, "occurrences": occs}
        for kw, occs in kw_occ_after.items()
        if len({o[1] for o in occs}) > 1
    ]

    report = {
        "total_rules": len(rules),
        "missing_code_before_count": len(missing_code_before),
        "missing_code_fixed_count": len(missing_code_fixed),
        "missing_code_unfixed_count": len(missing_code_unfixed),
        "keyword_conflicts_before_count": len(keyword_conflicts_before),
        "keyword_conflicts_after_count": len(keyword_conflicts_after),
        "removed_keywords_count": len(removed_keywords),
        "missing_code_fixed": missing_code_fixed[:200],
        "missing_code_unfixed": missing_code_unfixed[:200],
        "keyword_conflicts_before": keyword_conflicts_before[:200],
        "keyword_conflicts_after": keyword_conflicts_after[:200],
        "removed_keywords": removed_keywords[:400],
    }

    if output_json:
        Path(output_json).write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    if report_json:
        Path(report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def main():
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    root = Path(__file__).resolve().parents[1]
    category_csv = str(root / "data" / "category.csv")
    rule_base_json = str(root / "data" / "rule_base.json")
    out_json = str(root / "data" / "rule_base.synced.json")
    report_json = str(root / "docs" / "rule_base_sync_report.json")

    report = sync_rule_base_with_categories(
        category_csv=category_csv,
        rule_base_json=rule_base_json,
        output_json=out_json,
        report_json=report_json,
    )

    print("\n=== rule_base sync report ===")
    print(f"total_rules: {report['total_rules']}")
    print(f"missing_code_fixed: {report['missing_code_fixed_count']}")
    print(f"missing_code_unfixed: {report['missing_code_unfixed_count']}")
    print(f"keyword_conflicts_before: {report['keyword_conflicts_before_count']}")
    print(f"keyword_conflicts_after: {report['keyword_conflicts_after_count']}")
    print(f"removed_keywords: {report['removed_keywords_count']}")
    print(f"\nWrote: {out_json}")
    print(f"Wrote: {report_json}")


if __name__ == "__main__":
    main()

