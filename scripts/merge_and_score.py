from pathlib import Path
"""
Merge after-inference pickles from all three models, apply corrected matching,
compute metrics, write baseline_results.xlsx.

Corrected matching rule: for gold spans that contain a LAY_DICT key,
match detected entities against the normalised form of the gold span
and use the normalised target's core noun.
"""
import ast, re, pickle, warnings
import pandas as pd
warnings.filterwarnings("ignore")

GAP_FILE  = Path(__file__).resolve().parent.parent / "data" / "annotated" / "gap_reviewed.xlsx"
OUT_FILE  = Path(__file__).resolve().parent.parent / "results" / "baseline_results.xlsx"
PKL_SCI   = Path(__file__).resolve().parent.parent / "data" / "interim" / "after_ents_sci.pkl"
PKL_D4    = Path(__file__).resolve().parent.parent / "data" / "interim" / "after_ents_d4data.pkl"
PKL_SAM   = Path(__file__).resolve().parent.parent / "data" / "interim" / "after_ents_samrawal.pkl"

# LAY_DICT (70 lay-to-clinical mappings) is kept private because its keys are
# expressions taken from patient posts. See README (Data Access).
from lay_dict_private import LAY_DICT
LAY_KEYS_SORTED = sorted(LAY_DICT.keys(), key=len, reverse=True)

def normalise(text):
    for key in LAY_KEYS_SORTED:
        text = re.sub(re.escape(key), LAY_DICT[key], text, flags=re.IGNORECASE)
    return text

def get_after_target(gold_span):
    return normalise(gold_span)

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
    target = get_after_target(gold_span)
    cn = core_noun(target)
    return any(bidir_overlap(e, target) and cn in e.lower() for e in ent_list)

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
total_gold = len(df)

pkls = {}
for name, path in [("sci", PKL_SCI), ("d4data", PKL_D4), ("samrawal", PKL_SAM)]:
    try:
        with open(path, "rb") as f:
            pkls[name] = pickle.load(f)
        print(f"Loaded {name}: {len(pkls[name])} posts")
    except FileNotFoundError:
        print(f"MISSING pkl for {name} — skipping")

before_cols = {"sci":"sci_verdict_reviewed","d4data":"d4data_verdict_reviewed",
               "samrawal":"samrawal_verdict_reviewed"}
before_ent_cols = {"sci":"sci_all_entities","d4data":"d4data_all_entities",
                   "samrawal":"samrawal_all_entities"}

# ── span-level ────────────────────────────────────────────────────────────────
span_rows = []
for _, row in df.iterrows():
    pid  = row["post_id"]
    span = str(row["span_text"]) if pd.notna(row["span_text"]) else ""
    rec  = {"post_id": pid, "span_text": span,
            "clinical_concept": row.get("clinical_concept",""),
            "register": row.get("register",""),
            "match_target_after": get_after_target(span)}
    for mk in ["sci","d4data","samrawal"]:
        rec[f"caught_before_{mk}"] = (row.get(before_cols[mk],"miss") == "pass") if mk in pkls else False
        rec[f"caught_after_{mk}"]  = span_caught_after(pkls[mk].get(pid,[]), span) if mk in pkls else False
    span_rows.append(rec)

detail_df = pd.DataFrame(span_rows)

# ── summary ───────────────────────────────────────────────────────────────────
def postlevel_before(df, col):
    total = 0
    for _, grp in df.groupby("post_id"):
        post_ents = set()
        for val in grp[col].dropna():
            post_ents.update(e.lower() for e in parse_entities(val))
        total += len(post_ents)
    return total

def postlevel_after(ents_dict):
    return sum(len(set(e.lower() for e in v if e)) for v in ents_dict.values())

def f1(p, r): return 2*p*r/(p+r) if (p+r) else 0.0

prev = {
    "sci":     (0.4460, 0.4773, 0.2256, 0.2493, 0.2996, 0.3275,  3.12),
    "d4data":  (0.1733, 0.3125, 0.0789, 0.1103, 0.1084, 0.1631, 13.92),
    "samrawal":(0.5966, 0.5000, 0.5211, 0.3851, 0.5563, 0.4351, -9.66),
}

summary_rows = []
print("\n=== CORRECTED RESULTS (fixed after-matching) ===")

for mk in ["sci","d4data","samrawal"]:
    if mk not in pkls:
        summary_rows.append({"system": mk, "note": "pkl missing"})
        continue

    tp_b = detail_df[f"caught_before_{mk}"].sum()
    tp_a = detail_df[f"caught_after_{mk}"].sum()
    rb   = tp_b / total_gold
    ra   = tp_a / total_gold
    tb   = postlevel_before(df, before_ent_cols[mk])
    ta   = postlevel_after(pkls[mk])
    pb   = tp_b / tb if tb else 0.0
    pa   = tp_a / ta if ta else 0.0
    gain = (ra - rb) * 100

    pv = prev.get(mk, (0,0,0,0,0,0,0))
    changed = []
    if abs(ra - pv[1]) > 0.0005: changed.append(f"recall_after {pv[1]:.4f}→{ra:.4f}")
    if abs(pa - pv[3]) > 0.0005: changed.append(f"prec_after {pv[3]:.4f}→{pa:.4f}")
    if abs(gain - pv[6]) > 0.05:  changed.append(f"gain {pv[6]:.2f}→{gain:.2f} pp")

    print(f"\n{mk}:")
    print(f"  Before — TP={tp_b}/{total_gold}, recall={rb:.4f}, prec={pb:.4f}, F1={f1(pb,rb):.4f}")
    print(f"  After  — TP={tp_a}/{total_gold},  recall={ra:.4f},  prec={pa:.4f},  F1={f1(pa,ra):.4f}")
    print(f"  Gain:   {gain:+.2f} pp")
    if changed: print(f"  ** CHANGED: {', '.join(changed)}")
    else:       print(f"  (unchanged from previous run)")

    summary_rows.append({
        "system": mk,
        "recall_before":    round(rb, 4),
        "recall_after":     round(ra, 4),
        "precision_before": round(pb, 4),
        "precision_after":  round(pa, 4),
        "f1_before":        round(f1(pb,rb), 4),
        "f1_after":         round(f1(pa,ra), 4),
        "recall_gain_pp":   round(gain, 2),
    })

summary_df = pd.DataFrame(summary_rows)

with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Sheet1_Summary",    index=False)
    detail_df.to_excel(writer,  sheet_name="Sheet2_SpanDetail", index=False)

print(f"\nWritten: {OUT_FILE}")
print("\n=== FINAL TABLE 2 ===")
cols = ["system","recall_before","recall_after","precision_before",
        "precision_after","f1_before","f1_after","recall_gain_pp"]
print(summary_df[[c for c in cols if c in summary_df.columns]].to_string(index=False))
