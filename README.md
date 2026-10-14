# End-to-End Banking Intent Classification and Human-in-the-Loop Routing API

An automated customer support ticket triage and routing microservice powered by fine-tuned DistilBERT and dynamic INT8 post-training quantization. The service classifies customer banking inquiries across 5 specialized departments and enforces a confidence-threshold decision boundary to automatically route high-confidence tickets while triaging low-confidence edge cases to human specialists.

---

## External Resources and Documentation

- Google Drive (Dataset and Model Artifacts):
  https://drive.google.com/drive/folders/1QOEN8MGGfCfUd1WmtwMnR-yBopFZjjMF
- Google Colab Training Notebook:
  https://colab.research.google.com/drive/11gEVZKFhW9kuRiLTD7eLhQUc559z1Nsm#scrollTo=k8Rxf_eOSCcp
- Project Performance and Architecture Documentation:
  https://docs.google.com/document/d/1iD6ArN--5a12zk_pU3nERDgjRNR4aca1L52WRcx_Jgg/edit?tab=t.0

---

## Table of Contents

1. Core Objectives
2. Theoretical Concepts and Preprocessing Mechanics
3. Model Evolution: Baseline vs. Deep Learning
4. Training Dynamics and Optimization Setup
5. Comprehensive Model Performance Comparison
6. Unseen Test Set Evaluation (3,080 Tickets)
7. Production Human-in-the-Loop (HITL) Routing Simulation
8. Post-Training Quantization (INT8)
9. Repository Structure and Artifact Persistence
10. Installation and Local Setup
11. Running the FastAPI Inference Service
12. API Specifications and Payload Examples
13. End-to-End Latency Benchmarking
14. System Limitations and Next Steps

---

## 1. Core Objectives

- Develop a Statistical Baseline: Implement a classical machine learning pipeline (TF-IDF + Logistic Regression) to establish performance benchmarks, inference latency, and computational footprint.
- Fine-Tune a Transformer Architecture: Fine-tune `distilbert-base-uncased` to capture contextual nuances, complex sentence syntax, and subtle semantic distinctions across banking domains.
- Persist Production Artifacts: Decouple numerical weights (`model.safetensors`), architectural blueprints (`config.json`), and subword vocabularies to guarantee reproducible cold-start inference without retraining.
- Implement Production-Grade HITL Routing: Evaluate the model against a holdout test split of 3,080 tickets and introduce a confidence threshold mechanism ($\tau = 0.85$) that automates high-confidence queries while escalating ambiguous queries to human specialists.
- Optimize CPU Inference: Apply dynamic INT8 quantization to cut artifact storage by ~48% and deliver sub-100ms response times for real-time web services.

---

## 2. Theoretical Concepts and Preprocessing Mechanics

### A. Subword Tokenization and Tensor Construction

Classical NLP representations fail on out-of-vocabulary terms, typos, and banking jargon. Transformer-based tokenization mitigates this via WordPiece subword segmentation:

- Vocabulary Indexing (`input_ids`): Incoming text strings are broken into subwords and mapped to integer indices from a fixed 30,522 token vocabulary.
- Structural Control Tokens:
  - `[CLS]` (ID: 101): Positioned at index 0 of every token sequence. The final hidden state corresponding to this token aggregates bidirectional attention context across the entire sequence to serve as the sentence embedding for classification.
  - `[SEP]` (ID: 102): Appended to denote the sequence termination boundary.
- Padding and Truncation: Deep learning matrix operations require uniform rectangular batch tensors.
  - Inputs exceeding `max_length=64` are truncated.
  - Sequences shorter than 64 tokens are padded with zero tokens (`0`).
  - Analysis showed over 99% of customer support queries contain fewer than 50 tokens, ensuring zero truncation information loss while keeping tensor memory allocations minimal.
- Attention Masking (`attention_mask`): A binary mask array (`1` for genuine user tokens, `0` for padding tokens) instructing the multi-head self-attention mechanisms to assign zero mathematical weight to padding tokens during forward passes.

### B. Evaluation Metrics for Imbalanced Datasets

- Precision (False Alarm Rate): `True Positives / (True Positives + False Positives)`. Measures how often the model is correct when predicting a target department. Crucial to prevent misallocating tickets to specialized staff.
- Recall (Catch Rate): `True Positives / (True Positives + False Negatives)`. Measures the proportion of actual issues detected. Critical for security-critical classes like `fraud_security`, where false negatives cause financial loss.
- F1-Score: The harmonic mean of precision and recall `(2 * Precision * Recall) / (Precision + Recall)`. Unlike the arithmetic mean, the harmonic mean drops sharply if either metric degrades.
- Macro F1: Computes the unweighted mean of F1-scores across all 5 classes, providing equal evaluation weight to high-volume inquiries and minority classes (such as fraud).

---

## 3. Model Evolution: Baseline vs. Deep Learning

### Phase 1: Statistical Baseline (TF-IDF + Logistic Regression)

- Feature Extraction: Term Frequency-Inverse Document Frequency (TF-IDF) using unigram and bigram word groupings ($n \in \{1, 2\}$). Measures term importance within individual tickets penalised by corpus-wide frequency.
- Classification Layer: Logistic Regression optimizing a multi-class multinomial loss via Softmax.
- Strengths: Rapid training cycle (< 10 seconds), lightweight disk footprint (~5 MB), and sub-5ms CPU latency.
- Limitations: Ignores word ordering, syntax, and contextual negation (e.g., struggles to distinguish "charge on my card" from "card has no charge").
- Baseline Metrics: 87.0% Accuracy | 0.800 Macro F1.

### Phase 2: Deep Learning Transfer Learning (DistilBERT)

- Architecture: `distilbert-base-uncased`. Utilizes knowledge distillation to compress BERT-base by 40% (down to 66 million parameters) while retaining ~97% of language comprehension capacity and running 60% faster.
- Bidirectional Self-Attention: Simultaneously weighs tokens in both directions to resolve polysemy, syntactic dependencies, and intent nuance.
- Hardware Acceleration: Fine-tuned on an NVIDIA Tesla T4 GPU utilizing mixed precision (`fp16=True`) to minimize VRAM utilization and accelerate tensor computation.

---

## 4. Training Dynamics and Optimization Setup

- Optimizer: AdamW with weight decay
- Learning Rate: 5e-5
- Batch Size: 32
- Training Epochs: 3
- Max Sequence Length: 64

### Convergence Dynamics

| Epoch | Validation Loss | Validation Accuracy | Validation Macro F1 |
|---|:---:|:---:|:---:|
| 1 | 0.228 | 91.8% | 0.855 |
| 2 | 0.168 | 94.3% | 0.899 |
| 3 | 0.153 | 95.3% | 0.913 |

Monotonic decreases in both training and validation loss verified steady convergence with no evidence of overfitting.

---

## 5. Comprehensive Model Performance Comparison

| Attribute | Baseline Pipeline | Deep Learning Pipeline | Delta | Operational Impact |
|---|---|---|---|---|
| Algorithm | TF-IDF + Logistic Regression | Fine-Tuned DistilBERT | - | Semantic attention replaces bag-of-words |
| Accuracy | 87.0% | 95.3% (Val) / 95.0% (Test) | +8.3% (abs) | Error rate reduced by more than half |
| Macro F1 | 0.800 | 0.913 (Val) / 0.920 (Test) | +0.113 (abs) | +15% relative improvement on rare classes |
| Model Footprint | ~5 MB | ~255 MB (FP32) / ~138 MB (INT8) | - | Higher semantic fidelity at acceptable memory cost |
| Inference Latency | < 5ms (CPU) | ~20ms (GPU) / ~20ms (INT8 CPU) | - | Suitable for real-time customer support routing |
| Routing Strategy | Hard categorical guess | Softmax Confidence Routing | - | Isolates ambiguous ~4% queries for human intervention |

---

## 6. Unseen Test Set Evaluation (3,080 Tickets)

The final model was evaluated on a completely unobserved holdout test split of 3,080 customer queries:

| Department Class | Precision | Recall | F1-Score | Support (Volume) |
|---|:---:|:---:|:---:|:---:|
| card_issues | 0.89 | 0.91 | 0.90 | 360 |
| fraud_security | 0.91 | 0.92 | 0.91 | 160 |
| general_enquiry | 0.97 | 0.96 | 0.97 | 2,280 |
| login_access | 0.94 | 0.94 | 0.94 | 80 |
| transfers | 0.89 | 0.92 | 0.91 | 200 |
| **Macro Average** | **0.92** | **0.93** | **0.92** | **3,080** |
| **Overall Accuracy** | - | - | **0.95 (95.0%)** | **3,080** |

### Key Test Set Findings

- Out-of-Sample Generalization: Test Macro F1 (0.920) closely tracked validation performance (0.913), validating that the model generalized without memorizing training phrases.
- Minority Class Robustness: Despite `login_access` accounting for only 2.6% (80 tickets) and `fraud_security` accounting for 5.2% (160 tickets) of test volume, the model recorded strong F1-scores of 0.94 and 0.91 respectively.

---

## 7. Production Human-in-the-Loop (HITL) Routing Simulation

To prevent unconstrained automated misclassifications, predictions are evaluated through a confidence boundary:

- Decision Rule:
  - If `max P(intent | query) >= 0.85`: Execute `AUTO_ROUTE` (`escalate_to_human = false`).
  - If `max P(intent | query) < 0.85`: Triage to `HUMAN_ESCALATION` (`escalate_to_human = true`).

### Operational Distribution at Threshold tau = 0.85

- Total Holdout Inquiries: 3,080
- Automated Hands-Free Tickets (Confidence >= 0.85): 2,961 tickets (96.1% of total volume)
- Escalated to Human Review (Confidence < 0.85): 119 tickets (3.9% of total volume)
- Accuracy on Automated Stream: 96.59%

This threshold configuration eliminates 96% of repetitive manual triage work while filtering the ambiguous 4% of queries directly to experienced human agents.

---

## 8. Post-Training Quantization (INT8)

To optimize the model for CPU-based containerized deployments, dynamic INT8 quantization was applied to the PyTorch linear projection layers:

```bash
python quantize.py
```

### Quantization Results

- Disk Footprint: Reduced by 48.2% (from 255.4 MB down to 132.3 MB).
- Latency Reduction: 1.5x to 2.1x CPU speedup compared to FP32 CPU execution.
- Agreement Rate: > 99% classification output agreement with the original FP32 model.

The production runtime wrapper in `app/model.py` automatically detects and prioritizes the quantized weights (`weights_quantized/model_quantized.pt`) when executing on CPU environments.

---

## 9. Repository Structure and Artifact Persistence

```text
banking-classifier-api/
├── app/
│   ├── __init__.py          # Module initialization
│   ├── main.py              # FastAPI application, routing logic, and lifespan management
│   ├── model.py             # Inference pipeline wrapper, tokenization, and warm-up
│   └── schemas.py           # Pydantic schemas for request validation and response serialisation
├── weights/                 # Original FP32 fine-tuned model and tokenizer configuration
│   ├── config.json          # Transformer architecture definition
│   ├── tokenizer.json       # Subword vocabulary encoding rules
│   └── tokenizer_config.json
├── weights_quantized/       # INT8 dynamic quantized model artifact
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── benchmark.py             # Latency benchmarking script (100 HTTP requests)
├── quantize.py              # Dynamic quantization execution and accuracy validation script
├── requirements.txt         # Production Python dependencies
├── .gitignore               # Git exclusions (weights, virtual environments, caches)
└── README.md                # Technical project documentation
```

---

## 10. Installation and Local Setup

### Prerequisites

- Python 3.9 or higher
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Dideoluwa/banking-classifier-api.git
cd banking-classifier-api
```

### 2. Configure Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Model Weights

Binary weights are hosted externally on Google Drive to maintain repository efficiency:

Google Drive Folder:
https://drive.google.com/drive/folders/1QOEN8MGGfCfUd1WmtwMnR-yBopFZjjMF

Download the files and place them into the designated directories:
- `weights/model.safetensors`
- `weights_quantized/model_quantized.pt`

If only `weights/model.safetensors` is downloaded, generate the quantized model locally:
```bash
python quantize.py
```

---

## 11. Running the FastAPI Inference Service

Start the web server using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The service loads the model into memory, executes a warm-up forward pass, and begins listening on port 8000.

Interactive Swagger API documentation:
`http://localhost:8000/docs`

Alternative ReDoc interface:
`http://localhost:8000/redoc`

---

## 12. API Specifications and Payload Examples

### 1. Health Verification

- Method: `GET`
- Path: `/health`

Response:
```json
{
  "status": "healthy",
  "device": "cpu",
  "threshold": 0.85
}
```

### 2. Intent Classification and Routing

- Method: `POST`
- Path: `/predict`
- Content-Type: `application/json`

#### Automated Routing Example (High Confidence >= 0.85)

Request:
```json
{
  "ticket_id": "TCK-8921",
  "text": "There is an unauthorized transaction of $400 on my checking account."
}
```

Response:
```json
{
  "ticket_id": "TCK-8921",
  "intent": "fraud_security",
  "confidence": 0.9842,
  "route": "AUTO_ROUTE",
  "escalate_to_human": false,
  "all_scores": {
    "card_issues": 0.0051,
    "fraud_security": 0.9842,
    "general_enquiry": 0.0032,
    "login_access": 0.0041,
    "transfers": 0.0034
  }
}
```

#### Human Escalation Example (Low Confidence < 0.85)

Request:
```json
{
  "ticket_id": "TCK-4012",
  "text": "I need help with my paper documents and something went wrong."
}
```

Response:
```json
{
  "ticket_id": "TCK-4012",
  "intent": "general_enquiry",
  "confidence": 0.6214,
  "route": "HUMAN_ESCALATION",
  "escalate_to_human": true,
  "all_scores": {
    "card_issues": 0.0812,
    "fraud_security": 0.0521,
    "general_enquiry": 0.6214,
    "login_access": 0.1143,
    "transfers": 0.1310
  }
}
```

---

## 13. End-to-End Latency Benchmarking

With the server running on port 8000, execute the benchmarking utility:

```bash
python benchmark.py
```

The script transmits 100 sequential HTTP requests with representative banking payloads and records latency metrics:
- Mean Latency: ~18-30 ms on INT8 CPU
- Median Latency (p50): ~20 ms
- 95th Percentile (p95): ~35 ms
- SLA Target: Sub-100 ms target fully met.

---

## 14. System Limitations and Next Steps

1. Monolingual Scope: The current model operates on English queries (`uncased`). Expanding to multi-lingual customer bases will require fine-tuning multilingual models like XLM-RoBERTa.
2. Multi-Intent Formulation: Customers occasionally combine inquiries (e.g., reporting a stolen card while requesting a wire balance). Upgrading to a multi-label sigmoid classification head will support concurrent routing.
3. Active Learning Feedback Loop: Tickets routed to human review can be audited and relabeled by customer service specialists, generating high-value hard negative examples for iterative model fine-tuning.
4. Database Auditing: Integrating asynchronous request/response database logging (SQLite/PostgreSQL) to capture drift and query volume over time.
