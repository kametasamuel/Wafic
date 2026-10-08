"""
llm_core_noun_check.py
-----------------------
Applies a core-noun review to LLM (Claude and GPT-4o) true-positive detections,
analogous to the manual core-noun review applied to NER system outputs in the paper.

The paper defines the core symptom noun as "the head noun of the reference span:
the word that names the symptom rather than modifying or localising it."

Since dependency parsing is unavailable in this environment, a rule-based
first-content-word heuristic is used: skip leading adjectives, negations,
determiners, auxiliaries, and prepositions; take the first remaining word as
the core noun candidate. This approximates the head noun for the noun-phrase-
leading symptom spans that dominate the corpus (e.g., "heavy bleeding" -> "bleeding",
"pelvic pain" -> "pain", "abdominal distension" -> "distension").

Limitations:
  - Compound gold spans containing two symptoms (e.g., "not having appetite
    when I have pain") may have ambiguous core nouns; the heuristic takes the
    first content word ("appetite"), which may miss trailing co-occurring symptoms.
  - Trailing-element drops (e.g., "Heavy bleeding with clots" -> "Heavy bleeding")
    are not caught by first-noun heuristic; these are noted separately as they
    differ qualitatively from the BIO-tagging fragmentation mechanism.

Outputs:
  - Flagged pairs (gold span / detected span / inferred core noun) per model
  - Count of genuine vs. heuristic-artifact flags after manual inspection notes
"""

import pandas as pd
import ast

# ---------------------------------------------------------------------------
# Words to skip when looking for the first content noun in a symptom span.
# These are the leading function words, auxiliaries, and modifiers that
# precede the head noun in this corpus's symptom expressions.
# ---------------------------------------------------------------------------
SKIP_LEADING = {
    # personal pronouns and dialectal variants (gold spans often include
    # the subject, e.g. pregnancy comparisons or Pidgin pain expressions)
    'i', 'me', 'we', 'you', 'she', 'he', 'they', 'it',
    'mi', 'mia',  # dialectal "my"
    # determiners / possessives / quantifiers
    'a', 'an', 'the', 'my', 'her', 'his', 'their', 'our', 'its', 'this',
    'that', 'these', 'those', 'some', 'any', 'all', 'both', 'each', 'every',
    'lots', 'lot', 'plenty', 'bunch', 'another', 'other',
    # indefinite placeholders (not clinically specific)
    'something', 'anything', 'everything', 'nothing',
    # negations
    'no', 'not', 'never', 'nor', 'dont', "don't", 'didnt', "didn't",
    "can't", "cant", "won't", "wont", "isn't", "isnt",
    # auxiliaries and copulas
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did',
    'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',
    'get', 'getting', 'got', 'gotten',
    'feel', 'feeling', 'felt',
    'become', 'became', 'becoming',
    'look', 'looks', 'looked', 'looking',
    'seem', 'seems', 'seemed',
    'turn', 'turned', 'turning',
    'start', 'started', 'starting', 'begin', 'began', 'beginning',
    'keep', 'kept', 'keeping',
    'make', 'makes', 'made', 'making',
    'go', 'going', 'went', 'gone',
    'come', 'coming', 'came',
    'unable', 'able',
    # adverbs
    'very', 'quite', 'rather', 'extremely', 'so', 'too', 'also', 'even',
    'still', 'already', 'just', 'only', 'really', 'always', 'often',
    'sometimes', 'frequently', 'constantly', 'no longer', 'longer',
    'too', 'now', 'then', 'soon', 'again', 'daily', 'monthly', 'weekly',
    'seriously', 'completely', 'totally', 'absolutely', 'literally',
    # common leading adjectives in symptom spans
    'heavy', 'severe', 'serious', 'constant', 'terrible', 'painful',
    'bad', 'intense', 'extreme', 'mild', 'chronic', 'acute', 'sharp',
    'dull', 'excessive', 'frequent', 'irregular', 'abnormal', 'large',
    'big', 'small', 'little', 'much', 'many', 'more', 'most', 'less',
    # prepositions / conjunctions
    'in', 'on', 'at', 'with', 'during', 'of', 'from', 'for', 'to', 'by',
    'about', 'into', 'through', 'after', 'before', 'since', 'until',
    'when', 'while', 'as', 'if', 'because', 'though', 'although', 'but',
    'and', 'or', 'nor',
    # losing / loosing used as auxiliary verb sense
    'losing', 'loosing',
    # adjective forms that appear as leading modifiers in this corpus
    'chronic', 'recurrent', 'unnecessary',
    # auxiliary-like uses of "used" ("i used to have")
    'used',
    # verbs that appear as first content word in complex gold spans
    'eat', 'eats', 'ate', 'eating',
    # container/episode nouns that wrap the real symptom
    'episode', 'episodes', 'case', 'cases', 'instance',
}


def infer_core_noun(span):
    """
    Return the first content word of a symptom span as a proxy for its head noun.
    Skips leading words in SKIP_LEADING. Falls back to the first word if all
    words are in the skip set.
    """
    words = span.lower().strip().split()
    for word in words:
        clean = word.strip('.,;:!?()"\'')
        if clean and clean not in SKIP_LEADING:
            return clean
    # fallback: first word
    return words[0].strip('.,;:!?()"\'') if words else span


def spans_match(detected: str, gold: str) -> bool:
    """
    Exact replication of llm_baseline.py spans_match:
    bidirectional 50% TOKEN overlap (i2b2 partial match).
    Threshold is MAX of the two ratios >= 0.5.
    """
    d_tokens = set(detected.lower().split())
    g_tokens = set(gold.lower().split())
    if not d_tokens or not g_tokens:
        return False
    overlap = len(d_tokens & g_tokens)
    ratio_d = overlap / len(d_tokens)
    ratio_g = overlap / len(g_tokens)
    return max(ratio_d, ratio_g) >= 0.5


# ---------------------------------------------------------------------------
# Load gold annotations
# ---------------------------------------------------------------------------
gold_df = pd.read_excel(
    '../data/annotated/symptom_annotation.xlsx'
)

# ---------------------------------------------------------------------------
# Run check for each model
# ---------------------------------------------------------------------------
models = [
    ('Claude', '../claude_ner_results.csv'),
    ('GPT-4o', '../gpt4o_ner_results.csv'),
]

for model_name, csv_path in models:
    df = pd.read_csv(csv_path)

    total_tp = 0
    flagged = []

    for _, row in df.iterrows():
        pid = row['post_id']
        try:
            gold_spans = ast.literal_eval(row['gold_spans']) if isinstance(row['gold_spans'], str) else []
            det_spans  = ast.literal_eval(row['detected_spans']) if isinstance(row['detected_spans'], str) else []
        except Exception:
            continue

        matched_gold = set()
        for det_span in det_spans:
            for i, gold_span in enumerate(gold_spans):
                if i not in matched_gold and spans_match(det_span, gold_span):
                    matched_gold.add(i)
                    total_tp += 1
                    core = infer_core_noun(gold_span)
                    if core.lower() not in det_span.lower():
                        flagged.append({
                            'post_id':    pid,
                            'gold':       gold_span,
                            'detected':   det_span,
                            'core_noun':  core,
                        })
                    break  # each detection matched at most once

    print(f"\n{'='*60}")
    print(f"{model_name}: {total_tp} TPs matched, {len(flagged)} flagged by core-noun check")
    if flagged:
        print("  Flagged cases (gold | detected | core noun):")
        for f in flagged:
            print(f"    '{f['gold']}' | '{f['detected']}' | core='{f['core_noun']}'")
    else:
        print("  No flagged cases -- core noun present in all matched detections.")

print("""
Note on heuristic: the first-content-word approach correctly identifies the
head noun for noun-phrase-leading spans (the majority in this corpus). It will
not catch trailing-element drops on compound spans such as
'Heavy bleeding with clots' -> 'Heavy bleeding', where the trailing element
('clots') is also clinically meaningful. These cases differ qualitatively from
the BIO-tagging token-level fragmentation documented for d4data and are logged
separately during manual review.
""")
