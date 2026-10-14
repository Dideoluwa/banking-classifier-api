# Executive Report: End-to-End Banking Intent Classification and Human-in-the-Loop Routing

---

## 1. Executive Summary

This report documents the research, architecture, fine-tuning, and production evaluation of an automated customer inquiry triage microservice. The service routes incoming retail banking support tickets across 5 core business departments (`card_issues`, `fraud_security`, `general_enquiry`, `login_access`, `transfers`) and enforces a confidence-threshold decision boundary ($\tau = 0.85$) to separate high-confidence automated routing from low-confidence human escalation.

---

## 2. Resource Links

- Google Drive Repository (Datasets, FP32 and INT8 Weights):
  https://drive.google.com/drive/folders/1QOEN8MGGfCfUd1WmtwMnR-yBopFZjjMF
- Google Colab Training and Experimentation Notebook:
  https://colab.research.google.com/drive/11gEVZKFhW9kuRiLTD7eLhQUc559z1Nsm#scrollTo=k8Rxf_eOSCcp
- Live Documentation:
  https://docs.google.com/document/d/1iD6ArN--5a12zk_pU3nERDgjRNR4aca1L52WRcx_Jgg/edit?tab=t.0

---

## 3. Core Objectives

1. Develop a Statistical Baseline: Implement a classical machine learning pipeline (TF-IDF + Logistic Regression) to establish performance benchmarks, inference latency, and computational footprint.
2. Fine-Tune a Transformer Architecture: Fine-tune `distilbert-base-uncased` to capture contextual nuances, syntax, and subtle semantic shifts across banking intents.
3. Persist Production Artifacts: Store and decouple numerical model weights (`model.safetensors`), architectural specifications (`config.json`), and subword vocabularies to enable instant cold-start deployment without retraining.
4. Production-Grade Evaluation and HITL Routing: Evaluate the model on 3,080 unseen test queries and implement confidence thresholding to automate high-confidence queries while routing ambiguous edge cases to human specialists.
5. CPU Optimization: Apply dynamic INT8 quantization to achieve sub-100ms CPU latency and reduce model disk storage by ~48%.

---

## 4. Theoretical Concepts and Preprocessing Mechanics

### A. Subword Tokenization and Tensor Construction
- Vocabulary Indexing (`input_ids`): Customer text is tokenized into subwords using WordPiece, mapped to integer IDs from a fixed ~30,522 token vocabulary.
- Structural Control Tokens:
  - `[CLS]` (ID: 101): Positioned at index 0 of every input. Its final hidden vector aggregates context across the entire sequence, serving as the sequence representation for classification.
  - `[SEP]` (ID: 102): Appended to designate the sequence boundary.
- Padding and Truncation: Deep learning hardware requires uniform rectangular tensors for batch matrix multiplications. Inputs exceeding `max_length=64` are truncated; shorter inputs are padded with zeros (`0`). Over 99% of customer support queries fall below 50 tokens, eliminating truncation loss.
- Attention Masking (`attention_mask`): Binary array (`1` for genuine customer tokens, `0` for padding tokens) instructing multi-head self-attention mechanisms to assign zero mathematical weight to padding tokens during forward passes.

### B. Evaluation Metrics for Imbalanced Datasets
- Precision: `TP / (TP + FP)` - Measures how often the model is correct when predicting an intent. Crucial to prevent overwhelming human escalation queues with false alarms.
- Recall: `TP / (TP + FN)` - Measures the proportion of actual issues detected. Critical for `fraud_security`, where missing a compromised account incurs direct financial liability.
- F1-Score: Harmonic mean of precision and recall `(2 * Precision * Recall) / (Precision + Recall)`.
- Macro F1: Unweighted arithmetic average of F1-scores across all individual classes, weighting high-volume general questions and rare fraud occurrences equally.

---

## 5. Model Evolution and Training Dynamics

### Phase 1: Statistical Baseline (TF-IDF + Logistic Regression)
- Feature Extraction: TF-IDF with unigrams and bigrams ($n \in \{1, 2\}$).
- Classification: Multinomial Logistic Regression via Softmax.
- Metrics: 87.0% Accuracy | 0.800 Macro F1.
- Analysis: Rapid training (< 10s) and small footprint (~5 MB), but unable to resolve word ordering, syntax, or contextual negation.

### Phase 2: Deep Learning Transfer Learning (DistilBERT)
- Model: `distilbert-base-uncased` (66M parameters; 40% smaller than BERT-base, 60% faster, retaining ~97% language understanding).
- Hardware: NVIDIA Tesla T4 GPU with mixed precision (`fp16=True`).
- Hyperparameters: AdamW optimizer, learning rate $5 \times 10^{-5}$, batch size 32, epochs 3.

### Training Convergence

| Epoch | Validation Loss | Validation Accuracy | Validation Macro F1 |
|---|:---:|:---:|:---:|
| 1 | 0.228 | 91.8% | 0.855 |
| 2 | 0.168 | 94.3% | 0.899 |
| 3 | 0.153 | 95.3% | 0.913 |

---

## 6. Performance Comparison

| Model Architecture | Accuracy | Macro F1 | Delta (Accuracy) | Delta (Macro F1) | Model Footprint | Latency (CPU) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline: TF-IDF + Logistic Regression** | 87.0% | 0.800 | Baseline | Baseline | ~5 MB | < 5 ms |
| **Fine-Tuned: DistilBERT (FP32)** | 95.3% (Val) / 95.0% (Test) | 0.913 (Val) / 0.920 (Test) | +8.3% (abs) | +0.113 (abs) | ~255 MB | ~60 ms |
| **Quantized: DistilBERT (INT8)** | 94.8% (Test) | 0.918 (Test) | +7.8% (abs) | +0.118 (abs) | ~132 MB | ~20 ms |

---

## 7. Unseen Test Set Performance (3,080 Tickets)

| Department Class | Precision | Recall | F1-Score | Support (Volume) |
|---|:---:|:---:|:---:|:---:|
| card_issues | 0.89 | 0.91 | 0.90 | 360 |
| fraud_security | 0.91 | 0.92 | 0.91 | 160 |
| general_enquiry | 0.97 | 0.96 | 0.97 | 2,280 |
| login_access | 0.94 | 0.94 | 0.94 | 80 |
| transfers | 0.89 | 0.92 | 0.91 | 200 |
| **Macro Average** | **0.92** | **0.93** | **0.92** | **3,080** |
| **Overall Accuracy** | - | - | **0.95 (95.0%)** | **3,080** |

---

## 8. Human-in-the-Loop (HITL) Routing Simulation

Production routing evaluates the top predicted class probability against confidence threshold $\tau = 0.85$:

- High Confidence ($\ge 0.85$): Auto-routed to department queue.
- Low Confidence ($< 0.85$): Triaged to human specialist queue.

### Operational Results (3,080 Holdout Inquiries)

- Automated Tickets: 2,961 (96.1% of total volume)
- Escalated to Human Review: 119 (3.9% of total volume)
- Accuracy on Automated Queue: 96.59%

---

## 9. Error Analysis and Failure Modes

1. Intent Boundary Ambiguity: Customer tickets combining multiple operational problems (e.g. "My card was stolen and someone initiated a wire transfer") distribute probability mass across `card_issues`, `fraud_security`, and `transfers`, intentionally triggering human escalation.
2. Short or Underspecified Inquiries: Queries such as "can't sign in" without context trigger intermediate confidence scores, successfully preventing incorrect automated actions.
3. Terminology Overlap: Inquiries regarding transfer fees occasionally overlap with general inquiry fee queries, handled reliably by the confidence gating mechanism.
