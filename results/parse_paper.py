#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract experiment parameters from Dish-TS paper HTML"""
import re

with open("/tmp/paper.html", "r", encoding="utf-8") as f:
    html = f.read()

# Strip HTML tags
text = re.sub(r"<[^>]+>", " ", html)
text = re.sub(r"[ \t]+", " ", text)

# Save plain text for searching
with open("/tmp/paper.txt", "w", encoding="utf-8") as f:
    f.write(text)

# Find and print key experimental setup sections
keywords = [
    "batchsize", "batch size", "batch_size",
    "learning rate", "learning_rate", "10^{-4}", "1e-4",
    "early stop", "early stopping",
    "Implementation Detail", "implementation detail", "experiment setup",
    "5.1", "5.2", "5.3",
    "lookback", "look back", "look_back",
    "horizon", "prediction length",
    "Figure 3",
    "prior knowledge guidance",
    "α is", "α =", "alpha is",
    "ℓ = 1", "l = 1", "ell = 1",
    "d_model", "n_heads", "d_ff", "num_heads",
    "dropout",
    "24, 48, 96", "24, 48", "96, 192", "96 192 336",
    "Informer", "Autoformer", "N-BEATS",
    "electricity", "ETTm2", "ETTh1", "Weather", "illness",
    "MSE", "MAE",
]

printed = set()
for kw in keywords:
    idx = 0
    while True:
        pos = text.lower().find(kw.lower(), idx)
        if pos == -1 or (pos, kw) in printed:
            break
        start = max(0, pos - 200)
        end = min(len(text), pos + 300)
        snippet = text[start:end]
        # Clean up
        snippet = re.sub(r"\s+", " ", snippet)
        print("="*70)
        print(f"[KW] {kw}")
        print(snippet[:500])
        print()
        printed.add((pos, kw))
        idx = pos + 1
        if idx > pos + 1:  # Only first few matches
            break
        if len(printed) > 60:
            break

print("\n" + "="*70)
print("=== TOTAL UNIQUE CONTEXTS FOUND:", len(printed))
print("="*70)
