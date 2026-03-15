# material-category-mapping-ai

Multilingual material category mapping system for procurement data.
Designed to replace a fully manual classification process that didn't scale.

---

## Background

~100K multilingual material master records being classified manually.
Korean, English, Vietnamese, Chinese, Japanese — all mixed together,
with no consistent classification standard across teams.
Data quality was too low to use for any meaningful analysis.

---

## Approach

Focus was on building something operationally sustainable, not just a one-off model.

**Step 1 — Preprocessing**
Multilingual material name cleaning and feature extraction.
Language detection → normalization → structured format for training.

**Step 2 — Classification model**
- Multilingual embedding-based text similarity classification
- Triplet Loss + Hard Negative Mining training architecture
- Rule-based classification + AI recommendation hybrid architecture

**Step 3 — Human-in-the-loop**
Designed so accuracy compounds as users give feedback.
Not a static model — built to keep improving through operation.

---

## Category structure
```
Level 1 (top category)
  └─ Level 2
       └─ Level 3
            └─ Level 4 (leaf category)
```

AI maps each material to the appropriate level within the hierarchy,
maintaining structural consistency across the category tree.

---

## Results

- Dataset: ~**100K** multilingual material records
- Classification accuracy: **80%+** within standardized category schema
- Manual classification process replaced with automated pipeline
- Human-in-the-loop design enables continuous accuracy improvement

---

## Stack

`Python` `PyTorch` `SentenceTransformers` `Multilingual Embeddings`
`Triplet Loss` `Hard Negative Mining` `pandas` `scikit-learn`

---

## Structure
```
material-category-mapping-ai/
├── data/               # Sample data structure (no real data)
├── models/             # Model definitions
├── utils/              # Preprocessing utilities
├── train.py            # Training
├── inference.py        # Inference
├── config.py           # Configuration
└── requirements.txt
```
