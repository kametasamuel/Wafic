from pathlib import Path
"""
Diagnose scispaCy precision collapse.
Prints:
  1. Total unique entities detected before normalisation
  2. Total unique entities detected after normalisation
  3. TP count before and after
  4. 5 examples of entities added after normalisation that were NOT in before-set,
     with the LAY_DICT substitution that appears in that post.
"""
import ast, re, pickle, warnings
import pandas as pd
warnings.filterwarnings("ignore")

GAP_FILE = Path(__file__).resolve().parent.parent / "data" / "annotated" / "gap_reviewed.xlsx"
PKL_SCI  = Path(__file__).resolve().parent.parent / "data" / "interim" / "after_ents_sci.pkl"

# LAY_DICT (70 lay-to-clinical mappings) is kept private because its keys are
# expressions taken from patient posts. See README (Data Access).
from lay_dict_private import LAY_DICT
LAY_KEYS_SORTED = sorted(LAY_DICT.keys(), key=len, reverse=True)

def normalise(text):
    for key in LAY_KEYS_SORTED:
        text = re.sub(re.escape(key), LAY_DICT[key], text, flags=re.IGNORECASE)
    return text

def parse_entities(val):
    if pd.isna(val): return []
    s = str(val).strip()
    if not s or s == "nan": return []
    try:
        r = ast.literal_eval(s)
        if isinstance(r, list):
            return [str(e).strip() for e in r if str(e).strip()]
    except: pass
    return [e.strip() for e in s.split(",") if e.strip()]

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_excel(GAP_FILE)
with open(PKL_SCI, "rb") as f:
    after_ents = pickle.load(f)

posts = df[["post_id","text"]].drop_duplicates("post_id").set_index("post_id")["text"].fillna("")
gold_spans = df.groupby("post_id")["span_text"].apply(list).to_dict()

# before: from sci_all_entities column in gap_reviewed
before_col = "sci_all_entities"
before_col_verdict = "sci_verdict_reviewed"

# ── 1 & 2: entity counts ──────────────────────────────────────────────────────
total_before = 0
total_after  = 0
for pid in posts.index:
    post_rows = df[df["post_id"] == pid]
    before_set = set()
    for val in post_rows[before_col].dropna():
        before_set.update(e.lower() for e in parse_entities(val))
    total_before += len(before_set)
    after_set = set(e.lower() for e in after_ents.get(pid, []) if e)
    total_after += len(after_set)

print(f"\n1. Total unique entities BEFORE normalisation: {total_before}")
print(f"2. Total unique entities AFTER  normalisation: {total_after}")
print(f"   Ratio: {total_after/total_before:.2f}x")

# ── 3: TP counts ──────────────────────────────────────────────────────────────
tp_before = (df[before_col_verdict] == "pass").sum()

STOPWORDS = {"the","a","an","my","i","is","are","was","were","of","in","on",
             "at","to","and","or","with","during","have","has","had","it","its","this","that"}

def core_noun(text):
    tokens = text.lower().split()
    content = [t for t in tokens if t not in STOPWORDS] or tokens
    return content[-1].strip(".,;:!?\"'")

def bidir_overlap(ent, target):
    e, g = set(ent.lower()), set(target.lower())
    if not e or not g: return False
    inter = len(e & g)
    return (inter / len(e) >= 0.5) and (inter / len(g) >= 0.5)

def span_caught_after(ent_list, gold_span):
    target = normalise(gold_span)
    cn = core_noun(target)
    return any(bidir_overlap(e, target) and cn in e.lower() for e in ent_list)

tp_after = 0
for _, row in df.iterrows():
    pid = row["post_id"]
    span = str(row["span_text"]) if pd.notna(row["span_text"]) else ""
    if span_caught_after(after_ents.get(pid, []), span):
        tp_after += 1

print(f"\n3. TP BEFORE: {tp_before}/{len(df)}")
print(f"   TP AFTER:  {tp_after}/{len(df)}")

# ── 4: examples of new entities added after normalisation ─────────────────────
print("\n4. Sample new entities detected AFTER normalisation (not in before-set):\n")

examples = []
for pid, text in posts.items():
    post_rows = df[df["post_id"] == pid]
    before_set = set()
    for val in post_rows[before_col].dropna():
        before_set.update(e.lower() for e in parse_entities(val))
    after_set = set(e for e in after_ents.get(pid, []) if e)
    new_ents = [e for e in after_set if e.lower() not in before_set]
    if not new_ents:
        continue
    # find which LAY_DICT keys triggered in this post
    triggered = [key for key in LAY_KEYS_SORTED if re.search(re.escape(key), str(text), re.IGNORECASE)]
    if not triggered:
        continue
    for ent in new_ents[:2]:
        examples.append({
            "post_id": pid,
            "new_entity": ent,
            "lay_keys_in_post": triggered[:3],
            "norm_snippet": normalise(str(text))[:200],
        })
    if len(examples) >= 10:
        break

for ex in examples[:5]:
    print(f"  post_id:      {ex['post_id']}")
    print(f"  new entity:   {ex['new_entity']}")
    print(f"  lay keys:     {ex['lay_keys_in_post']}")
    print(f"  norm snippet: {ex['norm_snippet'][:180]}")
    print()
