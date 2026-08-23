# Isolated Kazakh Sign Language Recognition — KRSL-20, ADAT Encoder

Classifies isolated signs from the KRSL-20 dataset (5 signers, 20 gloss classes) by
adapting the encoder half of **ADAT** — a transformer architecture originally
designed for continuous sign-to-text *translation* — into a standalone sign
*classifier*.

**[Run it live on Kaggle →](https://www.kaggle.com/code/rakhatdos/isolated-kazakh-sign-language-recognition-using-th)**
(this notebook is Kaggle-native — it reads from a Kaggle-mounted dataset, so it's
published here as a record of the approach and results rather than a clone-and-run repo)

## Results

| | |
|---|---|
| Test accuracy | 85.64% |
| Validation accuracy | 82.70% |
| Train/val gap | 2.94% |
| Training | 100 epochs, best weights restored from epoch 99 |

Per-class test accuracy ranges from 71–95% across the 20 gloss classes — see the
confusion matrix and per-class breakdown at the bottom of the notebook for where it
struggles (mostly near-synonym question-word signs, e.g. `_kak` vs `_kak_q`).

## Approach

1. **Data.** Keypoint sequences (not raw video) for 5 signers performing 20 KRSL
   glosses, hosted as a public Kaggle dataset.
2. **Vocabulary fix.** The original gloss/text index mappings had a bug that
   misaligned labels — Cell 3 rebuilds the mapping correctly before anything else
   touches the data.
3. **Architecture.** The ADAT paper's encoder (log-sparse self-attention,
   transformer-based) is fetched directly from the authors' repo and wrapped with a
   global-average-pool + dense-softmax classification head. The decoder half is
   fetched but intentionally unused — this is classification, not sequence
   generation.
4. **Training.** Stratified 70/15/15 split, per-feature normalization fit on train
   only, class-weighted loss for the imbalanced gloss distribution.
5. **Evaluation.** Validation and held-out test evaluation run independently (Cell 9
   reloads saved weights/normalization stats rather than depending on in-memory state
   from Cell 8), so either can be re-run standalone.

## Attribution

The encoder architecture (`encoder.py`, `layers.py`) is not my code — it's fetched at
runtime from the official ADAT implementation:

> Shahin, N. & Ismail, L. (2026). ADAT: Novel Time-Series-Aware Adaptive Transformer
> Architecture for Sign Language Translation. *Scientific Reports*.
> https://doi.org/10.1038/s41598-026-36293-9
> Repo: https://github.com/INDUCE-Lab/ADAT-Adaptive-Transformer-for-Sign-Language-Translation
> (GPL-3.0 — the repo's README text says CC BY 4.0, but the actual `LICENSE` file in
> the repo is GPL-3.0, so that's treated as authoritative here)

Everything from Cell 2 onward — the KRSL-20 data pipeline, the vocabulary-mapping fix,
wiring the encoder into a classification head, the training loop, and the evaluation —
is my own adaptation, not part of the original repo.

This repo doesn't vendor copies of the ADAT source files; the notebook pulls them
fresh from the upstream repo each run, so nothing GPL-licensed is redistributed here.
