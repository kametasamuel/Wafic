"""
Regenerate figures/fig10_error_taxonomy.png: categories of the 119 gold spans
missed by all three NER systems (Section 6.1, Table 5, Figure 3 of the paper).

The counts below are aggregate results from the manual categorisation of the
missed spans; no patient text is needed to draw the figure.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "figures" / "fig10_error_taxonomy.png"
TOTAL = 119

CATEGORIES = [  # (label, count, colour), largest first
    ("Everyday synonym",     74, "#2E6A97"),
    ("Implied mention",      10, "#E8424F"),
    ("Functional impact",    10, "#E07B39"),
    ("Informal spelling",     8, "#6B4C9A"),
    ("Pregnancy comparison",  7, "#F4A261"),
    ("Cultural phrase",       5, "#2F6B4F"),
    ("Multi-symptom phrase",  5, "#8C8C8C"),
]
assert sum(c[1] for c in CATEGORIES) == TOTAL

def main():
    cats = CATEGORIES[::-1]  # largest bar at the top
    fig, ax = plt.subplots(figsize=(10, 6.05), dpi=266)
    y = list(range(len(cats)))
    ax.barh(y, [c[1] for c in cats], color=[c[2] for c in cats], height=0.6, alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c[0]}\n(n={c[1]})" for c in cats], fontsize=12)
    for i, c in enumerate(cats):
        ax.text(c[1] + 0.6, i, f"{c[1]}  ({round(100 * c[1] / TOTAL)}%)",
                va="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of spans", fontsize=13)
    ax.set_xlim(0, 92)
    ax.set_title("Why Did All Three Systems Miss These Spans?\n"
                 f"({TOTAL} spans missed by all three systems)", fontsize=14)
    ax.grid(axis="both", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    plt.tight_layout()
    plt.savefig(OUT)
    print(f"Saved {OUT}")

if __name__ == "__main__":
    main()
