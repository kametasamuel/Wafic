"""
run_scispacy.py
Runs scispaCy en_core_sci_sm on all posts in Raw.xlsx.
For each gold annotation in symptom_annotation.xlsx, records:
  - scispacy_output: matched entity text if model span overlaps gold span, else MISS

Output: data/interim/scispacy_results.csv
"""

import pandas as pd
import spacy
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW_PATH = BASE / "data/raw/Raw.xlsx"
ANNOT_PATH = BASE / "data/annotated/symptom_annotation.xlsx"
OUT_PATH = BASE / "data/interim/scispacy_results.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── load model ───────────────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_sci_sm")
except OSError:
    raise SystemExit(
        "Model not found. Run:\n"
        "  pip install scispacy\n"
        "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
    )

# ── load data ─────────────────────────────────────────────────────────────────
corpus = pd.read_excel(RAW_PATH, dtype=str)
annot = pd.read_excel(ANNOT_PATH, dtype=str)

# Ensure numeric offset columns
annot["span_start"] = pd.to_numeric(annot["span_start"], errors="coerce")
annot["span_end"]   = pd.to_numeric(annot["span_end"],   errors="coerce")

# ── run model per post ────────────────────────────────────────────────────────
# post_entities: post_id -> list of (start, end, text) — used for offset matching
# post_all_entities: post_id -> comma-separated entity texts for the full post
post_entities: dict[str, list[tuple[int, int, str]]] = {}
post_all_entities: dict[str, str] = {}

for _, row in corpus.iterrows():
    pid  = str(row["post_id"]).strip()
    text = str(row["text"]) if pd.notna(row.get("text")) else ""
    doc  = nlp(text)
    ents = [(e.start_char, e.end_char, e.text) for e in doc.ents]
    post_entities[pid]     = ents
    post_all_entities[pid] = ", ".join(dict.fromkeys(e[2] for e in ents)) if ents else ""

# ── match entities to gold spans ──────────────────────────────────────────────
def find_match(pid: str, gold_start: float, gold_end: float) -> str:
    """Return entity text if overlap covers >=50% of BOTH gold span and entity span."""
    entities = post_entities.get(pid, [])
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

results = []
for _, row in annot.iterrows():
    pid = str(row["post_id"]).strip()
    gs  = row["span_start"]
    ge  = row["span_end"]
    if pd.isna(gs) or pd.isna(ge):
        output = "MISS"
    else:
        output = find_match(pid, float(gs), float(ge))
    results.append(output)

annot["scispacy_output"]    = results
annot["sci_all_entities"]   = annot["post_id"].apply(
    lambda pid: post_all_entities.get(str(pid).strip(), ""))
annot.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"Saved {len(annot)} rows -> {OUT_PATH}")

# Quick summary
n_miss = (annot["scispacy_output"] == "MISS").sum()
print(f"Caught: {len(annot) - n_miss} / {len(annot)}  |  Miss: {n_miss}")
