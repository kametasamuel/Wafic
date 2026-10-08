from pathlib import Path
"""Run scispaCy on normalised text, save entity lists to pickle."""
import re, pickle, warnings
import pandas as pd
warnings.filterwarnings("ignore")

GAP_FILE = Path(__file__).resolve().parent.parent / "data" / "annotated" / "gap_reviewed.xlsx"
OUT_PKL  = Path(__file__).resolve().parent.parent / "data" / "interim" / "after_ents_sci.pkl"

# LAY_DICT (70 lay-to-clinical mappings) is kept private because its keys are
# expressions taken from patient posts. See README (Data Access).
from lay_dict_private import LAY_DICT
LAY_KEYS_SORTED = sorted(LAY_DICT.keys(), key=len, reverse=True)

def normalise(text):
    for key in LAY_KEYS_SORTED:
        text = re.sub(re.escape(key), LAY_DICT[key], text, flags=re.IGNORECASE)
    return text

import spacy
nlp = spacy.load("en_core_sci_sm")
print("scispaCy loaded")

df = pd.read_excel(GAP_FILE)
posts = df[["post_id","text"]].drop_duplicates("post_id").set_index("post_id")["text"].fillna("")

result = {}
for i, (pid, text) in enumerate(posts.items()):
    if i % 20 == 0: print(f"  {i+1}/{len(posts)}")
    norm = normalise(str(text))
    result[pid] = [e.text.strip() for e in nlp(norm).ents if e.text.strip()]

with open(OUT_PKL, "wb") as f:
    pickle.dump(result, f)
print(f"Saved {OUT_PKL}")
