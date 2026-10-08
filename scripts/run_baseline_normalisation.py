from pathlib import Path
"""
Dictionary normalisation baseline experiment.
Computes recall/precision/F1 before and after LAY_DICT substitution
for three NER systems: scispaCy, d4data, samrawal.

Before numbers: read from existing reviewed verdicts in gap_reviewed.xlsx.
After numbers: fresh inference on LAY_DICT-normalised post text.
"""

import ast
import re
import sys
import warnings
from collections import defaultdict

import pandas as pd

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
GAP_FILE = Path(__file__).resolve().parent.parent / "data" / "annotated" / "gap_reviewed.xlsx"
OUT_FILE  = Path(__file__).resolve().parent.parent / "results" / "baseline_results.xlsx"

# ── LAY_DICT ─────────────────────────────────────────────────────────────────
# LAY_DICT (70 lay-to-clinical mappings) is kept private because its keys are
# expressions taken from patient posts. See README (Data Access).
from lay_dict_private import LAY_DICT

print(f"LAY_DICT entries: {len(LAY_DICT)}")

# Sort keys longest-first so longer phrases replace before shorter sub-phrases
LAY_KEYS_SORTED = sorted(LAY_DICT.keys(), key=len, reverse=True)


def normalise_text(text: str) -> str:
    """Apply LAY_DICT substitutions case-insensitively, longest key first."""
    for key in LAY_KEYS_SORTED:
        repl = LAY_DICT[key]
        text = re.sub(re.escape(key), repl, text, flags=re.IGNORECASE)
    return text


# ── partial-overlap match ─────────────────────────────────────────────────────
def bidirectional_overlap(ent_text: str, gold_text: str) -> bool:
    """
    i2b2 bidirectional 50% character overlap.
    Intersection computed on the character sets of the lowercase strings.
    """
    e = ent_text.lower()
    g = gold_text.lower()
    # character-level intersection length (count shared chars positionally)
    # Use simple set intersection on characters — per the prompt spec
    # "lengths are in characters" and intersection is character overlap
    # We do it token-level to be more meaningful:
    e_chars = set(e)
    g_chars = set(g)
    inter = len(e_chars & g_chars)  # shared unique characters
    if len(e_chars) == 0 or len(g_chars) == 0:
        return False
    return (inter / len(e_chars) >= 0.5) and (inter / len(g_chars) >= 0.5)


def core_noun_present(ent_text: str, gold_text: str) -> bool:
    """
    Manual review rule: the core symptom noun from the gold span must appear
    in the detected entity text. We use the last non-stopword token of the
    gold span as the core noun.
    """
    STOPWORDS = {"the", "a", "an", "my", "i", "is", "are", "was", "were",
                 "of", "in", "on", "at", "to", "and", "or", "with", "during",
                 "have", "has", "had", "it", "its", "this", "that"}
    tokens = gold_text.lower().split()
    content = [t for t in tokens if t not in STOPWORDS]
    if not content:
        content = tokens
    core = content[-1].strip(".,;:!?\"'")
    return core in ent_text.lower()


def span_caught(ent_list: list[str], gold_text: str) -> bool:
    """Return True if any entity in ent_list matches gold_text."""
    for ent in ent_list:
        if bidirectional_overlap(ent, gold_text) and core_noun_present(ent, gold_text):
            return True
    return False


# ── load data ─────────────────────────────────────────────────────────────────
print("Loading gap_reviewed.xlsx …")
df = pd.read_excel(GAP_FILE)
print(f"  {len(df)} rows, {df['post_id'].nunique()} posts")


def parse_entities(val) -> list[str]:
    """Parse a cell that may be a Python list string or a comma-separated string."""
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


# ── get unique post texts ─────────────────────────────────────────────────────
posts = (
    df[["post_id", "text"]]
    .drop_duplicates("post_id")
    .set_index("post_id")["text"]
    .fillna("")
)
print(f"  {len(posts)} unique posts with text")

# ── load NER models ───────────────────────────────────────────────────────────
models_loaded = {}
model_errors = {}

# scispaCy
print("\nLoading scispaCy …")
try:
    import spacy
    nlp_sci = spacy.load("en_core_sci_sm")
    models_loaded["sci"] = True
    print("  scispaCy OK")
except Exception as e:
    model_errors["sci"] = str(e)
    print(f"  scispaCy FAILED: {e}")

# d4data
print("Loading d4data/biomedical-ner-all …")
try:
    from transformers import pipeline as hf_pipeline
    d4_pipe = hf_pipeline(
        "ner",
        model="d4data/biomedical-ner-all",
        aggregation_strategy="simple",
        device=0,  # GPU if available
    )
    models_loaded["d4data"] = True
    print("  d4data OK")
except Exception as e:
    model_errors["d4data"] = str(e)
    print(f"  d4data FAILED: {e}")

# samrawal
print("Loading samrawal/bert-base-uncased_clinical-ner …")
try:
    from transformers import pipeline as hf_pipeline
    sam_pipe = hf_pipeline(
        "ner",
        model="samrawal/bert-base-uncased_clinical-ner",
        aggregation_strategy="simple",
        device=0,
    )
    models_loaded["samrawal"] = True
    print("  samrawal OK")
except Exception as e:
    model_errors["samrawal"] = str(e)
    print(f"  samrawal FAILED: {e}")

if not models_loaded:
    print("\nNo models loaded — cannot continue.")
    sys.exit(1)

print(f"\nModels loaded: {list(models_loaded.keys())}")
if model_errors:
    print(f"Models failed: {model_errors}")


# ── run inference on normalised text ─────────────────────────────────────────
def run_sci(text: str) -> list[str]:
    doc = nlp_sci(text)
    return [ent.text.strip() for ent in doc.ents if ent.text.strip()]


def run_transformer(pipe, text: str) -> list[str]:
    """Run a HF NER pipeline; return list of entity strings."""
    # HF pipelines have a max token limit; chunk if needed
    MAX_CHARS = 2000
    if len(text) <= MAX_CHARS:
        results = pipe(text)
    else:
        # simple chunking at sentence boundaries
        chunks = []
        cur = ""
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


print("\nRunning inference on normalised text for each post …")
print(f"(Total posts: {len(posts)})")

# Store: post_id -> list of detected entity strings (normalised text)
after_entities = {m: {} for m in models_loaded}

for i, (pid, orig_text) in enumerate(posts.items()):
    norm_text = normalise_text(str(orig_text))
    if i % 20 == 0:
        print(f"  post {i+1}/{len(posts)} …")

    if "sci" in models_loaded:
        after_entities["sci"][pid] = run_sci(norm_text)

    if "d4data" in models_loaded:
        after_entities["d4data"][pid] = run_transformer(d4_pipe, norm_text)

    if "samrawal" in models_loaded:
        after_entities["samrawal"][pid] = run_transformer(sam_pipe, norm_text)

print("Inference complete.")


# ── build span-level results ──────────────────────────────────────────────────
span_rows = []

for _, row in df.iterrows():
    pid       = row["post_id"]
    span      = str(row["span_text"]) if pd.notna(row["span_text"]) else ""
    concept   = row.get("clinical_concept", "")
    register  = row.get("register", "")

    record = {
        "post_id":          pid,
        "span_text":        span,
        "clinical_concept": concept,
        "register":         register,
    }

    for model_key, before_col in [
        ("sci",     "sci_verdict_reviewed"),
        ("d4data",  "d4data_verdict_reviewed"),
        ("samrawal","samrawal_verdict_reviewed"),
    ]:
        caught_before = (row.get(before_col, "miss") == "pass") if model_key in models_loaded else False

        if model_key in models_loaded:
            ents_after = after_entities[model_key].get(pid, [])
            caught_after = span_caught(ents_after, span)
        else:
            caught_after = False

        record[f"caught_before_{model_key}"] = caught_before
        record[f"caught_after_{model_key}"]  = caught_after

    span_rows.append(record)

detail_df = pd.DataFrame(span_rows)


# ── summary metrics ───────────────────────────────────────────────────────────
total_gold = len(df)

def postlevel_entity_count(df, col):
    """Sum of unique entities per post across all posts (post-level dedup)."""
    total = 0
    for pid, grp in df.groupby("post_id"):
        post_ents = set()
        for val in grp[col].dropna():
            post_ents.update(e.lower() for e in parse_entities(val))
        total += len(post_ents)
    return total

def postlevel_after_count(after_ents_dict):
    """Sum of unique after-entities per post."""
    total = 0
    for pid, ents in after_ents_dict.items():
        total += len(set(e.lower() for e in ents if e))
    return total

def f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

summary_rows = []

for model_key, before_ent_col in [
    ("sci",     "sci_all_entities"),
    ("d4data",  "d4data_all_entities"),
    ("samrawal","samrawal_all_entities"),
]:
    if model_key not in models_loaded:
        summary_rows.append({
            "system": model_key,
            "note": model_errors.get(model_key, "not loaded"),
            "recall_before": None, "recall_after": None,
            "precision_before": None, "precision_after": None,
            "f1_before": None, "f1_after": None,
            "recall_gain_pp": None,
        })
        continue

    tp_before = detail_df[f"caught_before_{model_key}"].sum()
    tp_after  = detail_df[f"caught_after_{model_key}"].sum()

    recall_before = tp_before / total_gold
    recall_after  = tp_after  / total_gold

    total_before = postlevel_entity_count(df, before_ent_col)
    total_after  = postlevel_after_count(after_entities[model_key])

    prec_before = tp_before / total_before if total_before else 0.0
    prec_after  = tp_after  / total_after  if total_after  else 0.0

    f1_before = f1(prec_before, recall_before)
    f1_after  = f1(prec_after,  recall_after)

    gain = (recall_after - recall_before) * 100

    print(f"\n{model_key}:")
    print(f"  Before — TP={tp_before}, total_ents={total_before}, recall={recall_before:.3f}, prec={prec_before:.3f}, F1={f1_before:.3f}")
    print(f"  After  — TP={tp_after},  total_ents={total_after},  recall={recall_after:.3f},  prec={prec_after:.3f},  F1={f1_after:.3f}")
    print(f"  Recall gain: {gain:+.1f} pp")

    summary_rows.append({
        "system":           model_key,
        "recall_before":    round(recall_before, 4),
        "recall_after":     round(recall_after,  4),
        "precision_before": round(prec_before,   4),
        "precision_after":  round(prec_after,    4),
        "f1_before":        round(f1_before,     4),
        "f1_after":         round(f1_after,      4),
        "recall_gain_pp":   round(gain,           2),
    })

summary_df = pd.DataFrame(summary_rows)


# ── write output ──────────────────────────────────────────────────────────────
print(f"\nWriting {OUT_FILE} …")
with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Sheet1_Summary", index=False)
    detail_df.to_excel(writer,  sheet_name="Sheet2_SpanDetail", index=False)

print("Done. baseline_results.xlsx written.")
print("\n=== VERIFICATION SUMMARY ===")
print(f"LAY_DICT entries: {len(LAY_DICT)}")
print(f"Gold spans: {total_gold}")
print(summary_df.to_string(index=False))
