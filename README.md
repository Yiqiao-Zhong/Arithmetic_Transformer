#Addition with Small Transformers



---

## Project Overview
The project explores training compact Transformer models from scratch to under addition problems under varying carry constraints, aiming at interpretability.

---

## Datasets

| Variant | Format | Carry Conditions | Example |
|---------|--------|------------------|---------|
| Plain – no‑carry | `a+b=c\n` | None | `123+456=579\n` |
| Reverse – no‑carry | `a+b=rev(c)\n` | None | `123+456=975\n` |
| Plain – carry @ Hundredths position | `a+b=c\n` | Carry into hundreds | `860+390=1250\n` |
| Plain – carry @ Hundredths, Units | `a+b=c\n` | Carry into hundreds & tens | – |
| Plain/Reverse – full carry | same | Carry anywhere | – |

Dataset sizes ranged from **2500 – 10,000** examples.

---

## Model Architecture

- Layers: 6 (default) / 2 (small)  
- Heads: 6 (default) / 2 (small)  
- Embedding dimension: 384  
- Positional encoding: learned  
- Vocabulary: digits `0‑9`, symbols `+`, `=`, `\n`, `pad`

---

## Training Setup
- **Objective**  Causal‑LM (next‑token prediction)  
- **Optimizer**  AdamW (`β₁ = 0.9`, `β₂ = 0.99`, weight‑decay scheduling)  
- **Drop‑out**  0.20  
- **Batch size**  256  
- **Epochs**  14 (unless noted)

---

## Key Results

| Task | Direction | Test Accuracy |
|------|-----------|---------------|
| No‑carry | Forward | 100 % (3,000 ex) |
| No‑carry | Reverse | 100 % (3,000 ex) |
| Carry @ 100s | Forward | 90 % (2,500 ex) → **100 %** (5,000 ex) |
| Carry @ 100s | Reverse | 91 % (2,500 ex) → **100 %** (5,000 ex) |
| Carry @ 100s+10s | Forward | 98 % (5,000 ex) |
| Carry @ 100s+10s | Reverse | **100 %** (7,500 ex) |
| Full‑carry | Forward | 91 % (5,000 ex) → **99 %** (10,000 ex) |
| Full‑carry | Reverse | 96 % (5,000 ex) → **100 %** (10,000 ex) |

Reverse targets systematically need fewer examples/params for the same accuracy.

---

## Interpretability Insights
- **Error locus**  Most mistakes coincide with the first digit that requires a carry.  
- **2‑digit probe**  Padding with a leading `0` on 2-digit examples give correct answers for the first two digits.  
- **Embedding drift**  Digit tokens form structured clusters over epochs, mirroring arithmetic relations.  
- **Operator swap**  Feeding `a=b+\n` cripples accuracy → both token and positional embeddings matter.

---
## Images from Experiments

![Epoch Loss Graph](./extracted_images/train_accuracy.png)
![Epoch Accuracy Graph](./extracted_images/test_accuracy.png)
![Embedding Analysis](./extracted_images/epoch_14.png)

*(More visualisations live in `extracted_images/`.)*

---

## Currently working on:
1. Analysing the mistakes made by model during training.


---