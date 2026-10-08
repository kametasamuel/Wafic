# WAFIC-NER-Eval

Code and results for the paper:

**Symptom in Translation: Evaluating Clinical NLP Tools on West African Fibroid Patient Language**

## Overview

This repository contains the evaluation scripts, figures, and aggregated results for a study assessing three biomedical named entity recognition (NER) systems and two large language models on the West African Fibroid Informal Corpus (WAFIC), a manually annotated collection of 352 symptom spans from a public West African fibroid patient community.

## Repository Structure

```
wafic-ner-eval/
├── scripts/          # Evaluation and analysis scripts
├── figures/          # All paper figures (PNG)
└── results/          # Aggregated metrics (no patient text)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `run_scispacy.py` | Run scispaCy NER on the corpus |
| `run_transformer_ner.py` | Run d4data and samrawal transformer models |
| `run_baseline_normalisation.py` | LAY_DICT dictionary normalisation experiment |
| `llm_baseline.py` | Zero-shot LLM evaluation |
| `build_gap_table.py` | Merge system outputs with gold annotations |
| `merge_and_score.py` | Compute recall, precision, F1 with manual review |

## Results

`results/baseline_results.xlsx` reports recall, precision, and F1 for each NER system before and after dictionary normalisation, with no patient text included.

## Data Access

The WAFIC corpus contains patient-reported symptoms from a public online community. To protect community members, the corpus is not distributed with this repository. For the same reason, the 70-entry LAY_DICT used in the normalisation experiment is not included: its keys are expressions taken from patient posts. The scripts expect it in `scripts/lay_dict_private.py`, which is excluded from version control.

## Requirements

```bash
pip install -r requirements.txt
```

Note: scispaCy model (`en_core_sci_sm`) is included in `requirements.txt` and installs automatically.

## Citation

If you use this code or the WAFIC corpus in your work, please cite:

> [Citation to be added upon publication]
