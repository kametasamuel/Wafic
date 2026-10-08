from pathlib import Path
"""Run samrawal on normalised text, save entity lists to pickle."""
import re, pickle, warnings
import pandas as pd
warnings.filterwarnings("ignore")

GAP_FILE = Path(__file__).resolve().parent.parent / "data" / "annotated" / "gap_reviewed.xlsx"
OUT_PKL  = Path(__file__).resolve().parent.parent / "data" / "interim" / "after_ents_samrawal.pkl"

# LAY_DICT (70 lay-to-clinical mappings) is kept private because its keys are
# expressions taken from patient posts. See README (Data Access).
from lay_dict_private import LAY_DICT
LAY_KEYS_SORTED = sorted(LAY_DICT.keys(), key=len, reverse=True)

def normalise(text):
    for key in LAY_KEYS_SORTED:
        text = re.sub(re.escape(key), LAY_DICT[key], text, flags=re.IGNORECASE)
    return text

def run(pipe, text):
    MAX = 2000
    if len(text) <= MAX:
        return [r["word"].strip() for r in pipe(text) if r["word"].strip()]
    chunks, cur = [], ""
    for sent in text.split(". "):
        if len(cur) + len(sent) < MAX:
            cur += sent + ". "
        else:
            if cur: chunks.append(cur.strip())
            cur = sent + ". "
    if cur: chunks.append(cur.strip())
    out = []
    for c in chunks: out.extend(pipe(c))
    return [r["word"].strip() for r in out if r["word"].strip()]

from transformers import pipeline as hf_pipeline
pipe = hf_pipeline("ner", model="samrawal/bert-base-uncased_clinical-ner",
                   aggregation_strategy="simple", device=-1)
print("samrawal loaded")

df = pd.read_excel(GAP_FILE)
posts = df[["post_id","text"]].drop_duplicates("post_id").set_index("post_id")["text"].fillna("")

result = {}
for i, (pid, text) in enumerate(posts.items()):
    if i % 20 == 0: print(f"  {i+1}/{len(posts)}")
    result[pid] = run(pipe, normalise(str(text)))

with open(OUT_PKL, "wb") as f:
    pickle.dump(result, f)
print(f"Saved {OUT_PKL}")
