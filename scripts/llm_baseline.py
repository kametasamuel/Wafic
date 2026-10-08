"""
llm_baseline.py
LLM-based symptom extraction baseline using Claude and GPT-4o.
Evaluates against gold annotations from gap_reviewed.xlsx using
bidirectional 50% token overlap (i2b2 partial match).
"""

import os
import time
import json
import ast
import re
import pandas as pd
from collections import defaultdict

# Install dependencies if missing
try:
    import anthropic
except ImportError:
    os.system("pip install anthropic")
    import anthropic

try:
    import openai
except ImportError:
    os.system("pip install openai")
    import openai


DATA_PATH = "data/annotated/gap_reviewed.xlsx"
PROMPT_TEMPLATE = (
    "You are a clinical NLP assistant. Extract all symptom mentions from the "
    "following patient post. Return ONLY a JSON array of exact symptom spans as "
    "they appear in the text, e.g. [\"heavy bleeding\", \"pelvic pain\"]. Post: {text}"
)


# ---------------------------------------------------------------------------
# Token overlap matching (i2b2 partial match, bidirectional 50%)
# ---------------------------------------------------------------------------

def tokenize(span: str) -> set:
    return set(span.lower().split())


def spans_match(detected: str, gold: str) -> bool:
    """Return True if bidirectional 50% token overlap is satisfied."""
    d_tokens = tokenize(detected)
    g_tokens = tokenize(gold)
    if not d_tokens or not g_tokens:
        return False
    overlap = len(d_tokens & g_tokens)
    overlap_ratio_d = overlap / len(d_tokens)
    overlap_ratio_g = overlap / len(g_tokens)
    return max(overlap_ratio_d, overlap_ratio_g) >= 0.5


def evaluate(gold_spans: list, detected_spans: list) -> dict:
    """
    Match detected spans to gold spans (each gold matched at most once).
    Returns tp, fp, fn counts.
    """
    matched_gold = set()
    tp = 0
    fp = 0

    for det in detected_spans:
        matched = False
        for i, gold in enumerate(gold_spans):
            if i not in matched_gold and spans_match(det, gold):
                matched_gold.add(i)
                tp += 1
                matched = True
                break
        if not matched:
            fp += 1

    fn = len(gold_spans) - len(matched_gold)
    return {"tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str) -> list:
    """Extract JSON array from LLM response, handling markdown fences."""
    text = re.sub(r"```(?:json)?", "", response_text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            spans = json.loads(match.group())
            if isinstance(spans, list):
                return [str(s) for s in spans]
        except json.JSONDecodeError:
            pass
    try:
        spans = ast.literal_eval(text)
        if isinstance(spans, list):
            return [str(s) for s in spans]
    except Exception:
        pass
    return []


def call_claude(client, text: str) -> list:
    prompt = PROMPT_TEMPLATE.format(text=text)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text
    return parse_llm_response(raw)


def call_gpt4o(client, text: str) -> list:
    prompt = PROMPT_TEMPLATE.format(text=text)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    raw = response.choices[0].message.content
    return parse_llm_response(raw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_excel(DATA_PATH)

    required = {"post_id", "text", "span_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in spreadsheet: {missing}")

    # Build gold spans per post
    gold_by_post = defaultdict(list)
    text_by_post = {}
    for _, row in df.iterrows():
        pid = row["post_id"]
        gold_by_post[pid].append(str(row["span_text"]))
        text_by_post[pid] = str(row["text"])

    posts = [(pid, text_by_post[pid], gold_by_post[pid]) for pid in text_by_post]
    print(f"Found {len(posts)} unique posts.")

    # Init clients
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not anthropic_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")

    claude_client = anthropic.Anthropic(api_key=anthropic_key)
    gpt_client = openai.OpenAI(api_key=openai_key)

    claude_rows = []
    gpt_rows = []

    for i, (pid, text, gold_spans) in enumerate(posts):
        print(f"  [{i+1}/{len(posts)}] post_id={pid}", end=" ", flush=True)

        # --- Claude ---
        try:
            claude_detected = call_claude(claude_client, text)
        except Exception as e:
            print(f"\n    Claude error: {e}")
            claude_detected = []
        time.sleep(1)

        # --- GPT-4o ---
        try:
            gpt_detected = call_gpt4o(gpt_client, text)
        except Exception as e:
            print(f"\n    GPT-4o error: {e}")
            gpt_detected = []
        time.sleep(1)

        c_eval = evaluate(gold_spans, claude_detected)
        g_eval = evaluate(gold_spans, gpt_detected)

        claude_rows.append({
            "post_id": pid,
            "gold_spans": json.dumps(gold_spans),
            "detected_spans": json.dumps(claude_detected),
            "tp": c_eval["tp"],
            "fp": c_eval["fp"],
            "fn": c_eval["fn"],
        })
        gpt_rows.append({
            "post_id": pid,
            "gold_spans": json.dumps(gold_spans),
            "detected_spans": json.dumps(gpt_detected),
            "tp": g_eval["tp"],
            "fp": g_eval["fp"],
            "fn": g_eval["fn"],
        })

        print(f"| claude tp={c_eval['tp']} fp={c_eval['fp']} fn={c_eval['fn']} "
              f"| gpt4o tp={g_eval['tp']} fp={g_eval['fp']} fn={g_eval['fn']}")

    # Save CSVs
    claude_df = pd.DataFrame(claude_rows)
    gpt_df = pd.DataFrame(gpt_rows)
    claude_df.to_csv("claude_ner_results.csv", index=False)
    gpt_df.to_csv("gpt4o_ner_results.csv", index=False)
    print("\nSaved: claude_ner_results.csv, gpt4o_ner_results.csv")

    # Summary metrics
    def summarize(rows_df, name):
        tp = rows_df["tp"].sum()
        fp = rows_df["fp"].sum()
        fn = rows_df["fn"].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return {"Model": name, "TP": tp, "FP": fp, "FN": fn,
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4)}

    summary = pd.DataFrame([
        summarize(claude_df, "claude-sonnet-4-6"),
        summarize(gpt_df, "gpt-4o"),
    ])

    print("\n" + "=" * 60)
    print("SUMMARY (bidirectional 50% token overlap, i2b2 partial match)")
    print("=" * 60)
    print(summary.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
