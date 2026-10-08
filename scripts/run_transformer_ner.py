"""
run_transformer_ner.py
Runs TWO HuggingFace biomedical NER models on all posts in Raw.xlsx:
  - d4data/biomedical-ner-all       (biomedical, trained on PubMed)
  - samrawal/bert-base-uncased_clinical-ner  (clinical, trained on i2b2)

For each gold annotation, records per model:
  - <model>_output:       matched entity text (bidirectional >=50% overlap), or MISS
  - <model>_all_entities: all entity texts found in the full post, comma-separated

Output: data/interim/transformer_results.csv
"""

import gc
import pandas as pd
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

BASE       = Path(__file__).resolve().parent.parent
RAW_PATH   = BASE / "data/raw/Raw.xlsx"
ANNOT_PATH = BASE / "data/annotated/symptom_annotation.xlsx"
OUT_PATH   = BASE / "data/interim/transformer_results.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_CHARS = 2000
DEVICE    = 0 if torch.cuda.is_available() else -1

MODELS = [
    ("d4data",   "d4data/biomedical-ner-all"),
    ("samrawal", "samrawal/bert-base-uncased_clinical-ner"),
]

# ── load data ─────────────────────────────────────────────────────────────────
corpus = pd.read_excel(RAW_PATH, dtype=str)
annot  = pd.read_excel(ANNOT_PATH, dtype=str)
annot["span_start"] = pd.to_numeric(annot["span_start"], errors="coerce")
annot["span_end"]   = pd.to_numeric(annot["span_end"],   errors="coerce")

# ── helpers ───────────────────────────────────────────────────────────────────
def expand_to_word(text, s, e):
    """Expand entity span to full word boundaries."""
    while s > 0 and not text[s - 1].isspace() and text[s - 1] not in ",.!?;:()[]{}\"'":
        s -= 1
    while e < len(text) and not text[e].isspace() and text[e] not in ",.!?;:()[]{}\"'":
        e += 1
    return s, e

def find_match(entities, gold_start, gold_end):
    """Return entity text if overlap covers >=50% of BOTH gold span and entity span."""
    gold_len = gold_end - gold_start
    if gold_len <= 0:
        return "MISS"
    for e_start, e_end, e_text in entities:
        entity_len = e_end - e_start
        if entity_len <= 0:
            continue
        overlap = max(0, min(e_end, gold_end) - max(e_start, gold_start))
        if overlap / gold_len >= 0.5 and overlap / entity_len >= 0.5:
            return e_text
    return "MISS"

# ── run each model ────────────────────────────────────────────────────────────
def run_model(short_name, model_name):
    """Run one NER model over all posts. Returns (output_col, all_entities_col)."""
    print(f"\nLoading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForTokenClassification.from_pretrained(model_name)
    ner = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=DEVICE,
    )
    print(f"Loaded: {model_name}")

    post_entities:     dict[str, list] = {}
    post_all_entities: dict[str, str]  = {}

    total = len(corpus)
    for i, row in corpus.iterrows():
        pid  = str(row["post_id"]).strip()
        text = str(row["text"]) if pd.notna(row.get("text")) else ""
        text = text[:MAX_CHARS]

        try:
            preds = ner(text)
        except Exception as exc:
            print(f"  Warning: NER failed on {pid}: {exc}")
            preds = []

        entities = []
        for p in preds:
            if "start" not in p or "end" not in p:
                continue
            s, e = expand_to_word(text, p["start"], p["end"])
            entities.append((s, e, text[s:e]))

        post_entities[pid]     = entities
        post_all_entities[pid] = ", ".join(dict.fromkeys(e[2] for e in entities)) if entities else ""

        if (i + 1) % 20 == 0:
            print(f"  {short_name}: {i+1}/{total} posts...")

    # build per-annotation columns
    output_col      = []
    all_entities_col = []

    for _, row in annot.iterrows():
        pid = str(row["post_id"]).strip()
        gs, ge = row["span_start"], row["span_end"]
        if pd.isna(gs) or pd.isna(ge):
            output_col.append("MISS")
        else:
            output_col.append(find_match(post_entities.get(pid, []), float(gs), float(ge)))
        all_entities_col.append(post_all_entities.get(pid, ""))

    n_caught = sum(1 for v in output_col if v != "MISS")
    print(f"  {short_name} caught: {n_caught} / {len(annot)}")

    # free memory before next model
    del ner, model, tokenizer
    gc.collect()

    return output_col, all_entities_col

# ── run both models ───────────────────────────────────────────────────────────
results = {}
for short_name, model_name in MODELS:
    try:
        out, all_ents = run_model(short_name, model_name)
        results[short_name] = (out, all_ents)
    except Exception as exc:
        print(f"  FAILED to run {short_name}: {exc}")
        results[short_name] = (["MISS"] * len(annot), [""] * len(annot))

# ── save ──────────────────────────────────────────────────────────────────────
for short_name, (out, all_ents) in results.items():
    annot[f"{short_name}_output"]      = out
    annot[f"{short_name}_all_entities"] = all_ents

annot.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"\nSaved {len(annot)} rows -> {OUT_PATH}")
