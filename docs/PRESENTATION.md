# Executive Presentation: Automated Banking Ticket Triage & Human-in-the-Loop Routing

---

## Slide 1: Title and Problem Statement

### Title
**Automated Banking Intent Classification & Human-in-the-Loop Triage System**

### The Business Challenge
- High Inquiry Volume: Retail banks receive thousands of daily support tickets across card issues, fraud alerts, password lockouts, and wire transfers.
- Operational Bottleneck: Manual classification causes routing delays (hours to days), increases operational headcount costs, and risks misrouting time-sensitive security emergencies.
- Goal: Build a high-throughput, sub-100ms microservice that automatically routes high-confidence queries while safely escalating ambiguous edge cases to human specialists.

---

## Slide 2: Data Architecture & 5 Operational Intents

### Customer Inquiry Taxonomy
1. `card_issues`: Card activation, blocked debit cards, ATM card retention, card replacements.
2. `fraud_security`: Suspicious transactions, unrecognized charges, identity theft, phishing alerts.
3. `general_enquiry`: Branch opening hours, interest rates, fee structures, general inquiries.
4. `login_access`: Mobile app login failures, password resets, locked accounts, 2FA failures.
5. `transfers`: Wire transfers, international SWIFT/SEPA transfers, transfer limits, pending funds.

### The Real-World Challenge: Class Imbalance
- High-volume FAQ classes (`general_enquiry`: 74% of volume) vs. critical low-volume classes (`fraud_security`: ~5%, `login_access`: ~2.6%).
- Conventional accuracy is misleading: an AI could guess "general inquiry" 100% of the time and appear "accurate" while completely failing on fraud.

---

## Slide 3: Model Evolution & Engineering Strategy

### Phase 1: Statistical Baseline
- Architecture: TF-IDF (unigrams + bigrams) + Multinomial Logistic Regression.
- Characteristics: Fast training (< 10s), tiny footprint (~5 MB), < 5ms CPU latency.
- Weakness: Bag-of-words approach ignores word order and contextual nuances (e.g., cannot distinguish "charge on my card" from "card has no charge").
- Baseline Score: 87.0% Accuracy | 0.800 Macro F1.

### Phase 2: Deep Learning Transfer Learning
- Architecture: `distilbert-base-uncased` (66M parameters; 40% smaller, 60% faster than BERT-base).
- Hardware: Fine-tuned on NVIDIA Tesla T4 GPU with mixed precision (`fp16=True`).
- Setup: AdamW optimizer, learning rate 5e-5, batch size 32, 3 epochs.
- Validation Progression:
  - Epoch 1: Loss 0.228 | Acc 91.8% | Macro F1 0.855
  - Epoch 2: Loss 0.168 | Acc 94.3% | Macro F1 0.899
  - Epoch 3: Loss 0.153 | Acc 95.3% | Macro F1 0.913

### Phase 3: Post-Training Quantization (INT8)
- PyTorch dynamic INT8 quantization applied to linear projection layers.
- Results: 48.2% storage reduction (255 MB to 132 MB), ~2x CPU latency acceleration, >99% prediction agreement with FP32.

---

## Slide 4: Quantitative Results & Final Evaluation

### Holdout Test Set Performance (3,080 Unseen Tickets)

| Department Class | Precision | Recall | F1-Score | Support (Volume) |
|---|:---:|:---:|:---:|:---:|
| card_issues | 0.89 | 0.91 | 0.90 | 360 |
| fraud_security | 0.91 | 0.92 | 0.91 | 160 |
| general_enquiry | 0.97 | 0.96 | 0.97 | 2,280 |
| login_access | 0.94 | 0.94 | 0.94 | 80 |
| transfers | 0.89 | 0.92 | 0.91 | 200 |
| **Macro Average** | **0.92** | **0.93** | **0.92** | **3,080** |
| **Overall Accuracy** | - | - | **0.95 (95.0%)** | **3,080** |

### Key Takeaway
- Out-of-sample generalization verified: Test Macro F1 (0.920) matched validation (0.913).
- Minority class resilience: Achieved 0.91 F1 on `fraud_security` and 0.94 F1 on `login_access` despite low sample representation.

---

## Slide 5: Production Human-in-the-Loop (HITL) Routing

### The Confidence Gate Strategy
- Pure automation risks catastrophic misrouting when customers submit ambiguous queries.
- We enforce a Softmax confidence boundary: $\tau = 0.85$.
  - $\max P(\text{intent} \mid x) \ge 0.85 \longrightarrow$ **AUTO_ROUTE** (Hands-free automated dispatch).
  - $\max P(\text{intent} \mid x) < 0.85 \longrightarrow$ **HUMAN_ESCALATION** (Triaged to specialist staff).

### Operational Distribution (3,080 Holdout Inquiries)
- Automated Stream: 2,961 tickets (**96.1% of total volume**)
- Accuracy on Automated Stream: **96.59%**
- Escalated Stream: 119 tickets (**3.9% of total volume**)

### Business Value
Automates 96 out of every 100 customer tickets while isolating the ambiguous 4% before any customer experiences a misrouted ticket.

---

## Slide 6: Production Microservice & Architecture

```text
[ Customer Inquiries ]
         │
         ▼
[ FastAPI Inference Microservice (Port 8000) ]
   ├── Pydantic Input Validation
   ├── Subword Tokenizer (max_length=64)
   └── DistilBERT INT8 Engine (CPU Optimized)
         │
         ├──► HTTP JSON Response (< 30ms latency)
         │
         └──► [ Background Task ] ──► [ PostgreSQL Database ]
                                        - Prediction Audit Logs
                                        - Latency & Route Tracking
                                        - Live Monitoring (/stats, /logs)
```

- Decoupled Cold-Start: Model weights (`model.safetensors`, `model_quantized.pt`) load from local disk without retraining.
- Non-Blocking DB Logging: FastAPI `BackgroundTasks` write logs to PostgreSQL without adding latency to the client response.
- Containerized Reproducibility: Fully containerized with `Dockerfile` and `docker-compose.yml`.

---

## Slide 7: Error Analysis & Key Edge Cases

### Primary Failure Modes
1. Compound Inquiries: Customers asking multiple questions in one ticket (e.g., *"My card was stolen and I also need to send an international transfer"*).
   - Solution: Model confidence splits across classes, dropping below 0.85 and correctly triggering human triage.
2. Sparse Inquiries: Very brief tickets without context (e.g., *"Cannot sign in"*).
   - Solution: Gating mechanism isolates uncertain cases for human clarification.
3. Fee Overlaps: Distinguishing general fee inquiries from wire transfer fee schedules.

---

## Slide 8: Limitations & Future Roadmap

1. Multilingual Support: Extend architecture to multilingual representations (e.g., XLM-RoBERTa) to handle global customer bases.
2. Multi-Label Classification Head: Upgrade from Softmax single-label to Sigmoid multi-label classification to route compound tickets to multiple departments simultaneously.
3. Active Learning Loop: Route human-reviewed tickets directly into a continuous retraining dataset to automatically correct hard negative cases.
4. Model Performance Dashboard: Connect PostgreSQL prediction logs to Grafana or Metabase for real-time drift detection and SLA alerts.
