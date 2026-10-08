from pathlib import Path
"""
Samrawal-only inference on LAY_DICT-normalised text.
Reads existing baseline_results.xlsx for sci and d4data results,
adds samrawal after-normalisation numbers, overwrites the file.
"""

import ast
import re
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

GAP_FILE  = Path(__file__).resolve().parent.parent / "data" / "annotated" / "gap_reviewed.xlsx"
OUT_FILE  = Path(__file__).resolve().parent.parent / "results" / "baseline_results.xlsx"

# LAY_DICT (70 lay-to-clinical mappings) is kept private because its keys are
# expressions taken from patient posts. See README (Data Access).
from lay_dict_private import LAY_DICT

LAY_KEYS_SORTED = sorted(LAY_DICT.keys(), key=len, reverse=True)


def normalise_text(text: str) -> str:
    for key in LAY_KEYS_SORTED:
        text = re.sub(re.escape(key), LAY_DICT[key], text, flags=re.IGNORECASE)
    return text


def bidirectional_overlap(ent_text: str, gold_text: str) -> bool:
    e = set(ent_text.lower())
    g = set(gold_text.lower())
    inter = len(e & g)
    if not e or not g:
        return False
    return (inter / len(e) >= 0.5) and (inter / len(g) >= 0.5)


def core_noun_present(ent_text: str, gold_text: str) -> bool:
    STOPWORDS = {"the","a","an","my","i","is","are","was","were","of","in","on",
                 "at","to","and","or","with","during","have","has","had","it",
                 "its","this","that"}
    tokens = gold_text.lower().split()
    content = [t for t in tokens if t not in STOPWORDS] or tokens
    core = content[-1].strip(".,;:!?\"'")
    return core in ent_text.lower()


def span_caught(ent_list, gold_text):
    return any(
        bidirectional_overlap(e, gold_text) and core_noun_present(e, gold_text)
        for e in ent_list
    )


def parse_entities(val):
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s or s == "nan":
        return []
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return [str(e).strip() for e in result if str(e).strip()]
    except Exception:
        pass
    return [e.strip() for e in s.split(",") if e.strip()]


# ── load data ─────────────────────────────────────────────────────────────────
print("Loading gap_reviewed.xlsx …")
df = pd.read_excel(GAP_FILE)
total_gold = len(df)

posts = (
    df[["post_id", "text"]]
    .drop_duplicates("post_id")
    .set_index("post_id")["text"]
    .fillna("")
)
print(f"  {len(posts)} unique posts")

# ── load samrawal ─────────────────────────────────────────────────────────────
print("Loading samrawal/bert-base-uncased_clinical-ner …")
from transformers import pipeline as hf_pipeline
sam_pipe = hf_pipeline(
    "ner",
    model="samrawal/bert-base-uncased_clinical-ner",
    aggregation_strategy="simple",
    device=-1,  # CPU — avoids CUDA fragmentation on 8 GB GPU
)
print("  samrawal loaded OK")


def run_transformer(pipe, text):
    MAX_CHARS = 2000
    if len(text) <= MAX_CHARS:
        return [r["word"].strip() for r in pipe(text) if r["word"].strip()]
    chunks, cur = [], ""
    for sent in text.split(". "):
        if len(cur) + len(sent) < MAX_CHARS:
            cur += sent + ". "
        else:
            if cur:
                chunks.append(cur.strip())
            cur = sent + ". "
    if cur:
        chunks.append(cur.strip())
    results = []
    for chunk in chunks:
        results.extend(pipe(chunk))
    return [r["word"].strip() for r in results if r["word"].strip()]


# ── inference ─────────────────────────────────────────────────────────────────
print("Running samrawal on normalised text …")
sam_after = {}
for i, (pid, orig_text) in enumerate(posts.items()):
    if i % 20 == 0:
        print(f"  post {i+1}/{len(posts)} …")
    norm_text = normalise_text(str(orig_text))
    sam_after[pid] = run_transformer(sam_pipe, norm_text)
print("Inference done.")

# ── per-span results ──────────────────────────────────────────────────────────
sam_tp_before = (df["samrawal_verdict_reviewed"] == "pass").sum()

caught_after_col = []
for _, row in df.iterrows():
    pid  = row["post_id"]
    span = str(row["span_text"]) if pd.notna(row["span_text"]) else ""
    caught_after_col.append(span_caught(sam_after.get(pid, []), span))

sam_tp_after = sum(caught_after_col)

# precision after: unique entities per post, summed
sam_total_after = sum(len(set(e.lower() for e in ents if e)) for ents in sam_after.values())

# precision before: from existing column
def postlevel_dedup_count(df, col):
    total = 0
    for pid, grp in df.groupby("post_id"):
        post_ents = set()
        for val in grp[col].dropna():
            post_ents.update(e.lower() for e in parse_entities(val))
        total += len(post_ents)
    return total

sam_total_before = postlevel_dedup_count(df, "samrawal_all_entities")

recall_before  = sam_tp_before / total_gold
recall_after   = sam_tp_after  / total_gold
prec_before    = sam_tp_before / sam_total_before if sam_total_before else 0.0
prec_after     = sam_tp_after  / sam_total_after  if sam_total_after  else 0.0

def f1(p, r):
    return 2*p*r/(p+r) if (p+r) else 0.0

gain = (recall_after - recall_before) * 100

print(f"\nsamrawal:")
print(f"  Before — TP={sam_tp_before}, total_ents={sam_total_before}, recall={recall_before:.4f}, prec={prec_before:.4f}, F1={f1(prec_before,recall_before):.4f}")
print(f"  After  — TP={sam_tp_after},  total_ents={sam_total_after},  recall={recall_after:.4f},  prec={prec_after:.4f},  F1={f1(prec_after,recall_after):.4f}")
print(f"  Recall gain: {gain:+.1f} pp")

# ── merge into existing baseline_results.xlsx ─────────────────────────────────
print(f"\nUpdating {OUT_FILE} …")
existing_summary = pd.read_excel(OUT_FILE, sheet_name="Sheet1_Summary")
existing_detail  = pd.read_excel(OUT_FILE, sheet_name="Sheet2_SpanDetail")

# Replace samrawal row
sam_row = {
    "system":           "samrawal",
    "recall_before":    round(recall_before, 4),
    "recall_after":     round(recall_after,  4),
    "precision_before": round(prec_before,   4),
    "precision_after":  round(prec_after,    4),
    "f1_before":        round(f1(prec_before, recall_before), 4),
    "f1_after":         round(f1(prec_after,  recall_after),  4),
    "recall_gain_pp":   round(gain, 2),
}
existing_summary = existing_summary[existing_summary["system"] != "samrawal"]
new_summary = pd.concat([existing_summary[["system","recall_before","recall_after",
                                           "precision_before","precision_after",
                                           "f1_before","f1_after","recall_gain_pp"]],
                          pd.DataFrame([sam_row])], ignore_index=True)

# Add samrawal caught_after column to detail sheet
existing_detail["caught_after_samrawal"] = caught_after_col
# Also fill caught_before_samrawal from the reviewed verdict
existing_detail["caught_before_samrawal"] = (df["samrawal_verdict_reviewed"] == "pass").values

with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
    new_summary.to_excel(writer, sheet_name="Sheet1_Summary", index=False)
    existing_detail.to_excel(writer, sheet_name="Sheet2_SpanDetail", index=False)

print("Done. baseline_results.xlsx updated.")
print("\n=== FINAL TABLE 2 ===")
print(new_summary.to_string(index=False))
