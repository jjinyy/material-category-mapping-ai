import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _norm(s: str) -> str:
    return str(s or "").strip().lower()


def validate_rule_base(
    category_csv: str,
    rule_base_json: str,
    fail_on_missing_code: bool = False,
) -> dict:
    df_cat = pd.read_csv(category_csv, encoding="utf-8-sig")
    if df_cat.empty:
        raise ValueError(f"category.csv is empty: {category_csv}")
    if "CODE" not in df_cat.columns or "TYPE" not in df_cat.columns:
        raise ValueError("category.csv must include CODE and TYPE columns")

    code_to_type = dict(zip(df_cat["CODE"].astype(str), df_cat["TYPE"].astype(str)))
    codes = set(code_to_type.keys())

    rules = json.loads(Path(rule_base_json).read_text(encoding="utf-8"))
    if not isinstance(rules, list):
        raise ValueError("rule_base.json must be a JSON array (list)")

    missing_code = []
    type_mismatch = []
    malformed = []

    kw_to_codes = defaultdict(set)

    for i, r in enumerate(rules):
        if not isinstance(r, dict):
            malformed.append({"idx": i, "reason": "rule is not an object"})
            continue

        code = str(r.get("code", "")).strip()
        rtype = str(r.get("type", "")).strip()
        kws = r.get("keywords", [])

        if not code or not rtype or not isinstance(kws, list) or len(kws) == 0:
            malformed.append(
                {
                    "idx": i,
                    "reason": "missing required fields (code/type/keywords)",
                    "code": code,
                    "type": rtype,
                }
            )
            continue

        if code not in codes:
            missing_code.append({"idx": i, "code": code, "type": rtype, "keywords": kws[:10]})
        else:
            cat_type = code_to_type.get(code)
            if cat_type and cat_type != rtype:
                type_mismatch.append(
                    {
                        "idx": i,
                        "code": code,
                        "rule_type": rtype,
                        "category_type": cat_type,
                        "keywords": kws[:10],
                    }
                )

        for kw in kws:
            kw_norm = _norm(kw)
            if kw_norm:
                kw_to_codes[kw_norm].add(code)

    kw_conflicts = [
        {"keyword": kw, "codes": sorted(list(cs))}
        for kw, cs in kw_to_codes.items()
        if len(cs) > 1
    ]

    report = {
        "total_rules": len(rules),
        "missing_code_count": len(missing_code),
        "type_mismatch_count": len(type_mismatch),
        "malformed_count": len(malformed),
        "keyword_conflict_count": len(kw_conflicts),
        "missing_code": missing_code[:200],
        "type_mismatch": type_mismatch[:200],
        "malformed": malformed[:200],
        "keyword_conflicts": kw_conflicts[:200],
    }

    if fail_on_missing_code and report["missing_code_count"] > 0:
        raise RuntimeError(
            f"rule_base.json has {report['missing_code_count']} codes not in category.csv"
        )

    return report


def main():
    root = Path(__file__).resolve().parents[1]
    category_csv = str(root / "data" / "category.csv")
    rule_base_json = str(root / "data" / "rule_base.json")

    report = validate_rule_base(category_csv=category_csv, rule_base_json=rule_base_json)

    import sys

    # Windows console(cp949) 출력 깨짐/에러 방지
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("\n=== rule_base.json validation report ===")
    print(f"total_rules: {report['total_rules']}")
    print(f"missing_code_count: {report['missing_code_count']}")
    print(f"type_mismatch_count: {report['type_mismatch_count']}")
    print(f"malformed_count: {report['malformed_count']}")
    print(f"keyword_conflict_count: {report['keyword_conflict_count']}")

    if report["missing_code_count"]:
        print("\n[missing_code] examples:")
        for row in report["missing_code"][:20]:
            print(f"- idx={row['idx']} code={row['code']} type={row['type']} kw={row['keywords'][:3]}")

    if report["type_mismatch_count"]:
        print("\n[type_mismatch] examples:")
        for row in report["type_mismatch"][:20]:
            print(
                f"- idx={row['idx']} code={row['code']} rule_type={row['rule_type']} category_type={row['category_type']}"
            )

    if report["keyword_conflict_count"]:
        print("\n[keyword_conflicts] examples:")
        for row in report["keyword_conflicts"][:20]:
            kw = row["keyword"]
            try:
                print(f"- keyword='{kw}' -> {row['codes']}")
            except UnicodeEncodeError:
                kw_safe = kw.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
                print(f"- keyword='{kw_safe}' -> {row['codes']}")

    print("\nDone.")


if __name__ == "__main__":
    main()

