from pathlib import Path
"""
Corrected baseline normalisation experiment — all three models.

Fix: for after-normalisation evaluation, match detected entities against
the LAY_DICT replacement term (not the original gold span) for any gold
span that contains a LAY_DICT key. Core-noun check also uses the replacement.

Before numbers: read from existing reviewed verdicts (unchanged).
After numbers: fresh inference with corrected matching.
"""

import ast
import re
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

GAP_FILE = Path(__file__).resolve().parent.parent / "data" / "annotated" / "gap_reviewed.xlsx"
OUT_FILE = Path(__file__).resolve().parent.parent / "results" / "baseline_results.xlsx"

# LAY_DICT (70 lay-to-clinical mappings) is kept private because its keys are
# expressions taken from patient posts. See README (Data Access).
from lay_dict_private import LAY_DICT

LAY_KEYS_SORTED = sorted(LAY_DICT.keys(), key=len, reverse=True)


def normalise_text(text: str) -> str:
    for key in LAY_KEYS_SORTED:
        text = re.sub(re.escape(key), LAY_DICT[key], text, flags=re.IGNORECASE)
    return text


def get_after_match_target(gold_span: str) -> str:
    """
    Return the target string for after-normalisation matching.
    If the gold span contains a LAY_DICT key, return the normalised
    version of the gold span (i.e. apply LAY_DICT substitutions to it).
    If unchanged, return the original gold span.
    """
    normalised = normalise_text(gold_span)
    return normalised  # if no key matched, this equals gold_span


STOPWORDS = {"the","a","an","my","i","is","are","was","were","of","in","on",
             "at","to","and","or","with","during","have","has","had","it",
             "its","this","that"}


def core_noun(text: str) -> str:
    tokens = text.lower().split()
    content = [t for t in tokens if t not in STOPWORDS] or tokens
    return content[-1].strip(".,;:!?\"'")


def bidirectional_overlap(ent_text: str, target: str) -> bool:
    """Bidirectional 50% character-set overlap."""
    e = set(ent_text.lower())
    g = set(target.lower())
    if not e or not g:
        return False
    inter = len(e & g)
    return (inter / len(e) >= 0.5) and (inter / len(g) >= 0.5)


def span_caught_after(ent_list: list, gold_span: str) -> bool:
    """
    After-normalisation match: compare entities against the normalised
    form of the gold span; use normalised target's core noun.
    """
    target = get_after_match_target(gold_span)
    cn = core_noun(target)
    for ent in ent_list:
        if bidirectional_overlap(ent, target) and cn in ent.lower():
            return True
    return False


def parse_entities(val) -> list:
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
print(f"  {total_gold} gold spans, {df['post_id'].nunique()} posts")

posts = (
    df[["post_id", "text"]]
    .drop_duplicates("post_id")
    .set_index("post_id")["text"]
    .fillna("")
)

# ── load models ───────────────────────────────────────────────────────────────
models_loaded = {}
model_errors  = {}

print("\nLoading scispaCy (en_core_sci_sm) …")
try:
    import spacy
    nlp_sci = spacy.load("en_core_sci_sm")
    models_loaded["sci"] = True
    print("  OK")
except Exception as e:
    model_errors["sci"] = str(e)
    print(f"  FAILED: {e}")

print("Loading d4data/biomedical-ner-all …")
try:
    from transformers import pipeline as hf_pipeline
    d4_pipe = hf_pipeline(
        "ner",
        model="d4data/biomedical-ner-all",
        aggregation_strategy="simple",
        device=0,
    )
    models_loaded["d4data"] = True
    print("  OK")
except Exception as e:
    model_errors["d4data"] = str(e)
    print(f"  FAILED: {e}")

print("Loading samrawal/bert-base-uncased_clinical-ner …")
try:
    from transformers import pipeline as hf_pipeline
    sam_pipe = hf_pipeline(
        "ner",
        model="samrawal/bert-base-uncased_clinical-ner",
        aggregation_strategy="simple",
        device=-1,  # CPU — CUDA fragmentation on this GPU
    )
    models_loaded["samrawal"] = True
    print("  OK")
except Exception as e:
    model_errors["samrawal"] = str(e)
    print(f"  FAILED: {e}")

if not models_loaded:
    print("No models loaded.")
    sys.exit(1)

print(f"\nLoaded: {list(models_loaded.keys())}")


# ── inference helpers ─────────────────────────────────────────────────────────
def run_sci(text: str) -> list:
    return [ent.text.strip() for ent in nlp_sci(text) if ent.text.strip()]


def run_transformer(pipe, text: str) -> list:
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


# ── run after-normalisation inference ────────────────────────────────────────
print(f"\nRunning after-normalisation inference on {len(posts)} posts …")

after_ents = {m: {} for m in models_loaded}

for i, (pid, orig_text) in enumerate(posts.items()):
    if i % 20 == 0:
        print(f"  post {i+1}/{len(posts)} …")
    norm_text = normalise_text(str(orig_text))

    if "sci" in models_loaded:
        after_ents["sci"][pid] = run_sci(norm_text)
    if "d4data" in models_loaded:
        after_ents["d4data"][pid] = run_transformer(d4_pipe, norm_text)
    if "samrawal" in models_loaded:
        after_ents["samrawal"][pid] = run_transformer(sam_pipe, norm_text)

print("Inference complete.")


# ── build span-level detail ───────────────────────────────────────────────────
span_rows = []

for _, row in df.iterrows():
    pid    = row["post_id"]
    span   = str(row["span_text"]) if pd.notna(row["span_text"]) else ""
    record = {
        "post_id":          pid,
        "span_text":        span,
        "clinical_concept": row.get("clinical_concept", ""),
        "register":         row.get("register", ""),
        "match_target":     get_after_match_target(span),   # for audit
    }
    for mk, before_col in [("sci","sci_verdict_reviewed"),
                            ("d4data","d4data_verdict_reviewed"),
                            ("samrawal","samrawal_verdict_reviewed")]:
        record[f"caught_before_{mk}"] = (
            row.get(before_col, "miss") == "pass"
        ) if mk in models_loaded else False

        if mk in models_loaded:
            record[f"caught_after_{mk}"] = span_caught_after(
                after_ents[mk].get(pid, []), span
            )
        else:
            record[f"caught_after_{mk}"] = False

    span_rows.append(record)

detail_df = pd.DataFrame(span_rows)


# ── summary metrics ───────────────────────────────────────────────────────────
def postlevel_dedup_before(df, col):
    total = 0
    for _, grp in df.groupby("post_id"):
        post_ents = set()
        for val in grp[col].dropna():
            post_ents.update(e.lower() for e in parse_entities(val))
        total += len(post_ents)
    return total

def postlevel_dedup_after(ents_dict):
    return sum(len(set(e.lower() for e in v if e)) for v in ents_dict.values())

def f1(p, r):
    return 2*p*r/(p+r) if (p+r) else 0.0

# Previously reported numbers (for comparison)
prev = {
    "sci":     dict(recall_before=0.4460, recall_after=0.4773,
                    precision_before=0.2256, precision_after=0.2493,
                    f1_before=0.2996, f1_after=0.3275, recall_gain_pp=3.12),
    "d4data":  dict(recall_before=0.1733, recall_after=0.3125,
                    precision_before=0.0789, precision_after=0.1103,
                    f1_before=0.1084, f1_after=0.1631, recall_gain_pp=13.92),
    "samrawal":dict(recall_before=0.5966, recall_after=0.5000,
                    precision_before=0.5211, precision_after=0.3851,
                    f1_before=0.5563, f1_after=0.4351, recall_gain_pp=-9.66),
}

before_ent_cols = {
    "sci":     "sci_all_entities",
    "d4data":  "d4data_all_entities",
    "samrawal":"samrawal_all_entities",
}

summary_rows = []
print("\n=== CORRECTED RESULTS ===")

for mk in ["sci", "d4data", "samrawal"]:
    if mk not in models_loaded:
        summary_rows.append({"system": mk,
                              "note": model_errors.get(mk, "not loaded")})
        continue

    tp_before  = detail_df[f"caught_before_{mk}"].sum()
    tp_after   = detail_df[f"caught_after_{mk}"].sum()

    rec_before = tp_before / total_gold
    rec_after  = tp_after  / total_gold

    tot_before = postlevel_dedup_before(df, before_ent_cols[mk])
    tot_after  = postlevel_dedup_after(after_ents[mk])

    prec_before = tp_before / tot_before if tot_before else 0.0
    prec_after  = tp_after  / tot_after  if tot_after  else 0.0

    gain = (rec_after - rec_before) * 100

    changed = []
    p = prev.get(mk, {})
    if abs(rec_after - p.get("recall_after", 0)) > 0.0005:
        changed.append(f"recall_after {p.get('recall_after',0):.4f}→{rec_after:.4f}")
    if abs(prec_after - p.get("precision_after", 0)) > 0.0005:
        changed.append(f"prec_after {p.get('precision_after',0):.4f}→{prec_after:.4f}")
    if abs(gain - p.get("recall_gain_pp", 0)) > 0.05:
        changed.append(f"gain {p.get('recall_gain_pp',0):.2f}→{gain:.2f} pp")

    print(f"\n{mk}:")
    print(f"  Before — TP={tp_before}/{total_gold}, recall={rec_before:.4f}, prec={prec_before:.4f}, F1={f1(prec_before,rec_before):.4f}")
    print(f"  After  — TP={tp_after}/{total_gold},  recall={rec_after:.4f},  prec={prec_after:.4f},  F1={f1(prec_after,rec_after):.4f}")
    print(f"  Gain:   {gain:+.2f} pp")
    if changed:
        print(f"  ** CHANGED from previous run: {', '.join(changed)}")
    else:
        print(f"  (no change from previous run)")

    summary_rows.append({
        "system":           mk,
        "recall_before":    round(rec_before,  4),
        "recall_after":     round(rec_after,   4),
        "precision_before": round(prec_before, 4),
        "precision_after":  round(prec_after,  4),
        "f1_before":        round(f1(prec_before, rec_before), 4),
        "f1_after":         round(f1(prec_after,  rec_after),  4),
        "recall_gain_pp":   round(gain, 2),
    })

summary_df = pd.DataFrame(summary_rows)


# ── write output ──────────────────────────────────────────────────────────────
print(f"\nWriting {OUT_FILE} …")
with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Sheet1_Summary",   index=False)
    detail_df.to_excel(writer,  sheet_name="Sheet2_SpanDetail", index=False)

print("Done.")
print("\n=== FINAL TABLE 2 ===")
print(summary_df[["system","recall_before","recall_after","precision_before",
                   "precision_after","f1_before","f1_after","recall_gain_pp"]].to_string(index=False))
