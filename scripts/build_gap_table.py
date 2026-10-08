"""
build_gap_table.py
Merges scispaCy and transformer results (two transformer models) with the gold
annotation file, applies manual review decisions, and outputs the final gap table.

Inputs:
  data/interim/scispacy_results.csv
  data/interim/transformer_results.csv
  data/annotated/suspect_review.xlsx   (if reviewer decisions are present)

Output:
  data/annotated/gap_terms_categorized.xlsx

Column order:
  [annotation cols]
  scispacy_output  | sci_all_entities  | sci_verdict
  d4data_output    | d4data_all_entities    | d4data_verdict
  samrawal_output  | samrawal_all_entities  | samrawal_verdict
  tools_catching   (comma-separated list of tools that passed)
  gap_verdict      all_catch | two_catch | one_catch | neither
"""

import pandas as pd
from pathlib import Path

BASE         = Path(__file__).resolve().parent.parent
SCI_PATH     = BASE / "data/interim/scispacy_results.csv"
TRANS_PATH   = BASE / "data/interim/transformer_results.csv"
OUT_PATH     = BASE / "data/annotated/gap_terms_categorized.xlsx"
SUSPECT_PATH = BASE / "data/annotated/suspect_review.xlsx"

TOOLS = ["sci", "d4data", "samrawal"]

# ── load ──────────────────────────────────────────────────────────────────────
if not SCI_PATH.exists():
    raise FileNotFoundError(f"Run run_scispacy.py first — missing: {SCI_PATH}")
if not TRANS_PATH.exists():
    raise FileNotFoundError(f"Run run_transformer_ner.py first — missing: {TRANS_PATH}")

sci   = pd.read_csv(SCI_PATH,   dtype=str)
trans = pd.read_csv(TRANS_PATH, dtype=str)

# ── build base table ──────────────────────────────────────────────────────────
ORIG_COLS = [
    "post_id", "text", "span_text", "span_start", "span_end",
    "register", "mention_type", "clinical_concept", "notes",
]
orig_cols_present = [c for c in ORIG_COLS if c in sci.columns]
merged = sci[orig_cols_present].copy()

merged["scispacy_output"]       = sci["scispacy_output"]
merged["sci_all_entities"]      = sci.get("sci_all_entities", "")
merged["d4data_output"]         = trans.get("d4data_output",         "MISS")
merged["d4data_all_entities"]   = trans.get("d4data_all_entities",   "")
merged["samrawal_output"]       = trans.get("samrawal_output",       "MISS")
merged["samrawal_all_entities"] = trans.get("samrawal_all_entities", "")

# ── apply manual review decisions ────────────────────────────────────────────
# Review sheet columns: sci_decision, d4data_decision, samrawal_decision
# Values: confirm_catch | reclassify_miss
# Blank decisions on flagged rows default to confirm_catch.
OUTPUT_COLS = {
    "sci":     "scispacy_output",
    "d4data":  "d4data_output",
    "samrawal":"samrawal_output",
}

if SUSPECT_PATH.exists():
    review = pd.read_excel(SUSPECT_PATH, dtype=str)
    decision_cols = [f"{t}_decision" for t in TOOLS]
    has_decisions = any(
        c in review.columns and review[c].notna().any()
        for c in decision_cols
    )
    if has_decisions:
        for t in TOOLS:
            col = f"{t}_decision"
            if col not in review.columns:
                review[col] = "confirm_catch"
            review[col] = review[col].fillna("confirm_catch")

        join_key = ["post_id", "span_text", "scispacy_output", "d4data_output", "samrawal_output"]
        # only use keys present in review
        join_key = [k for k in join_key if k in review.columns]
        lookup = review.set_index(join_key)[[f"{t}_decision" for t in TOOLS]]

        def apply_decisions(row):
            key = tuple(str(row.get(k, "")).strip() for k in join_key)
            if key in lookup.index:
                decs = lookup.loc[key]
                if hasattr(decs, "iloc") and decs.ndim == 2:
                    decs = decs.iloc[0]
                for t in TOOLS:
                    if decs.get(f"{t}_decision", "confirm_catch") == "reclassify_miss":
                        row[OUTPUT_COLS[t]] = "MISS"
            return row

        merged = merged.apply(apply_decisions, axis=1)
        for t in TOOLS:
            n = (review[f"{t}_decision"] == "reclassify_miss").sum()
            if n:
                print(f"  {t}: {n} reclassified as MISS")

# ── verdict columns ───────────────────────────────────────────────────────────
for t, out_col in OUTPUT_COLS.items():
    merged[f"{t}_verdict"] = merged[out_col].apply(
        lambda x: "miss" if x == "MISS" else "pass")

def tools_catching(row):
    catching = [t for t in TOOLS if row[f"{t}_verdict"] == "pass"]
    return ", ".join(catching) if catching else ""

merged["tools_catching"] = merged.apply(tools_catching, axis=1)

def gap_verdict(row):
    n = sum(row[f"{t}_verdict"] == "pass" for t in TOOLS)
    if n == 3: return "all_catch"
    if n == 2: return "two_catch"
    if n == 1: return "one_catch"
    return "neither"

merged["gap_verdict"] = merged.apply(gap_verdict, axis=1)

# ── reorder columns ───────────────────────────────────────────────────────────
final_cols = orig_cols_present + [
    "scispacy_output",  "sci_all_entities",     "sci_verdict",
    "d4data_output",    "d4data_all_entities",   "d4data_verdict",
    "samrawal_output",  "samrawal_all_entities", "samrawal_verdict",
    "tools_catching",   "gap_verdict",
]
merged = merged[[c for c in final_cols if c in merged.columns]]

# ── suspect flagging / review export ─────────────────────────────────────────
def flag_suspect(caught, gold):
    if caught == "MISS":
        return False
    gold_len = len(str(gold).strip())
    return gold_len > 0 and len(str(caught).strip()) / gold_len < 0.6

for t, out_col in OUTPUT_COLS.items():
    merged[f"{t}_suspect"] = merged.apply(
        lambda r, c=out_col: flag_suspect(r[c], r["span_text"]), axis=1)

any_suspect = merged[[f"{t}_suspect" for t in TOOLS]].any(axis=1)

if SUSPECT_PATH.exists():
    review = pd.read_excel(SUSPECT_PATH, dtype=str)
    has_decisions = any(
        c in review.columns and review[c].notna().any()
        for c in [f"{t}_decision" for t in TOOLS]
    )
else:
    has_decisions = False

if not has_decisions:
    review_cols = (
        ["post_id", "span_text", "clinical_concept", "register"] +
        [c for pair in [(OUTPUT_COLS[t], f"{t}_suspect") for t in TOOLS] for c in pair] +
        ["gap_verdict", "notes"]
    )
    suspects = merged[any_suspect][[c for c in review_cols if c in merged.columns]].copy()
    for t in TOOLS:
        suspects[f"{t}_decision"] = ""
    suspects.to_excel(SUSPECT_PATH, index=False)
    counts = {t: merged[f"{t}_suspect"].sum() for t in TOOLS}
    print(f"\nSuspect rows exported: {len(suspects)}  {counts}")
    print(f"Fill *_decision columns with: confirm_catch  or  reclassify_miss")
    print(f"Review file -> {SUSPECT_PATH}")

# ── drop internal suspect flags before saving ─────────────────────────────────
merged = merged.drop(columns=[f"{t}_suspect" for t in TOOLS], errors="ignore")

# ── save ──────────────────────────────────────────────────────────────────────
merged.to_excel(OUT_PATH, index=False)
print(f"\nSaved {len(merged)} rows -> {OUT_PATH}")

# ── summary ───────────────────────────────────────────────────────────────────
total = len(merged)
print("\nGAP VERDICT (three-model):")
for v, n in merged["gap_verdict"].value_counts().items():
    print(f"  {v:<12} {n:>4}  ({100*n/total:.1f}%)")

print("\nPer-tool recall:")
for t, out_col in OUTPUT_COLS.items():
    n_pass = (merged[f"{t}_verdict"] == "pass").sum()
    print(f"  {t:<10} {n_pass:>4} / {total}  ({100*n_pass/total:.1f}%)")

print("\nBy register:")
print(
    merged.groupby(["register", "gap_verdict"])
    .size().unstack(fill_value=0).to_string()
)

print("\nBy mention_type:")
print(
    merged.groupby(["mention_type", "gap_verdict"])
    .size().unstack(fill_value=0).to_string()
)
