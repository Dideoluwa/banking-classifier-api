# Banking Ticket Triage and Intent Classifier API

An automated customer support ticket triage and routing microservice powered by a fine-tuned DistilBERT model. The service classifies incoming banking queries into specific department intents and applies a confidence-threshold decision gate to determine whether tickets are automatically routed or escalated to human customer support agents.

---

## Table of Contents

- Project Overview
- Dataset and Google Colab Notebook
- Architecture and Intent Classes
- Routing and Escalation Logic
- Project Structure
- Installation and Setup
- Model Weights Configuration
- Dynamic INT8 Quantization
- Running the API
- API Endpoints and Usage
- Latency Benchmarking
- Deliverables and Artifacts

---

## Project Overview

In retail banking customer support, handling inquiry routing manually introduces latency and routing errors. This project implements a complete machine learning inference service that:

1. Ingests raw customer support messages.
2. Tokenizes and processes inputs through a fine-tuned transformer classification model (DistilBERT).
3. Produces calibrated softmax probabilities across 5 core banking domains.
4. Evaluates output confidence against a predefined threshold (0.85).
5. Routes confident predictions automatically or flags low-confidence queries for human agent intervention.
6. Employs dynamic INT8 post-training quantization to reduce memory footprint and achieve sub-100ms CPU latency.

---

## Dataset and Google Colab Notebook

The complete training dataset, exploratory data analysis, and model fine-tuning workflow are available in the project Google Drive folder:

Google Drive Resources:
https://drive.google.com/drive/folders/1QOEN8MGGfCfUd1WmtwMnR-yBopFZjjMF

This drive contains:
- The labeled banking support ticket dataset.
- The Google Colab training notebook containing data preprocessing, baseline model training, DistilBERT fine-tuning, loss curves, and evaluation metrics.
- Exported model weights and artifacts.

---

## Architecture and Intent Classes

The classifier categorizes queries into 5 operational intents:

1. `card_issues`: Card activation, blocked cards, physical ATM card retention, replacement requests, and card transaction errors.
2. `fraud_security`: Suspicious account activity, unauthorized transactions, identity theft alerts, and phishing reports.
3. `general_enquiry`: Interest rates, branch opening hours, general fee schedules, and product eligibility.
4. `login_access`: Mobile app sign-in failures, two-factor authentication issues, locked accounts, and password resets.
5. `transfers`: Domestic and international wire transfers, SEPA/SWIFT inquiries, transaction limits, and pending transfer statuses.

---

## Routing and Escalation Logic

To balance automation throughput with operational risk, the inference engine evaluates the top softmax confidence score against a decision boundary:

- Confidence >= 0.85: `route = "AUTO_ROUTE"` and `escalate_to_human = false`.
- Confidence < 0.85: `route = "HUMAN_ESCALATION"` and `escalate_to_human = true`.

This ensures ambiguous customer requests or edge-case formulations are safely handed off to human staff rather than misrouted to the wrong banking department.

---

## Project Structure

```text
banking-classifier-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and endpoint definitions
│   ├── model.py             # Inference engine wrapper, warm-up, and routing logic
│   └── schemas.py           # Pydantic request and response schemas
├── weights/                 # Original FP32 fine-tuned DistilBERT weights & tokenizer
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── weights_quantized/       # INT8 dynamic quantized model artifact
│   ├── model_quantized.pt
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── benchmark.py             # HTTP latency benchmark script (100 sample requests)
├── quantize.py              # Dynamic quantization and FP32 vs INT8 benchmark script
├── requirements.txt         # Production dependencies
├── .gitignore               # Git exclusions (large weights, caches, virtual environments)
└── README.md                # Project documentation
```

---

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Dideoluwa/banking-classifier-api.git
cd banking-classifier-api
```

### 2. Create and Activate a Virtual Environment

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

---

## Model Weights Configuration

Large model binaries are hosted externally to adhere to repository size limits. Download the model weights from the provided Google Drive folder:

Google Drive: https://drive.google.com/drive/folders/1QOEN8MGGfCfUd1WmtwMnR-yBopFZjjMF

Place the downloaded files into their respective directories:
- `weights/model.safetensors`
- `weights_quantized/model_quantized.pt`

If only the original FP32 model is downloaded, the quantized version can be generated locally using `quantize.py`.

---

## Dynamic INT8 Quantization

The project includes an optimization script that applies PyTorch dynamic quantization to the model linear layers:

```bash
python quantize.py
```

### Quantization Benefits

- Storage Reduction: Reduces model size by approximately 48% (from 268 MB to 138 MB).
- Inference Acceleration: Provides approximately 1.5x to 2x latency reduction on standard CPU architectures.
- Accuracy Retention: Retains over 99% classification agreement relative to the unquantized FP32 model.

The inference service in `app/model.py` automatically checks for `weights_quantized/model_quantized.pt` on startup and prioritizes it for CPU execution.

---

## Running the API

Start the Uvicorn ASGI server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Upon startup, the service initializes the model wrapper, pre-warms the inference pipeline, and binds to port 8000.

Interactive Swagger API documentation is available at:
`http://localhost:8000/docs`

Alternative ReDoc documentation is available at:
`http://localhost:8000/redoc`

---

## API Endpoints and Usage

### 1. Health Check

Verifies server availability and model initialization.

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

### 2. Predict Intent and Route

Classifies support inquiry text and returns routing determinations.

- Method: `POST`
- Path: `/predict`
- Headers: `Content-Type: application/json`

Request Body:
```json
{
  "ticket_id": "TCK-8921",
  "text": "There is a fraudulent charge of $400 on my checking account from yesterday."
}
```

Response Body:
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

Low-Confidence Example (Human Escalation):

Request Body:
```json
{
  "ticket_id": "TCK-4012",
  "text": "I need help with my paper documents and something went wrong."
}
```

Response Body:
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

## Latency Benchmarking

To benchmark end-to-end API latency against the running server:

```bash
python benchmark.py
```

The script runs 100 sequential HTTP requests across diverse banking queries and calculates latency distribution metrics:
- Average end-to-end latency
- Median (p50)
- 95th Percentile (p95)
- Verification against the target SLA (< 100ms)

---

## Deliverables and Artifacts Summary

1. GitHub Repository: Clean, modular codebase structured for production deployment.
2. README: Documentation covering installation, configuration, testing, and operations.
3. Dataset and Notebook: Hosted on Google Drive (https://drive.google.com/drive/folders/1QOEN8MGGfCfUd1WmtwMnR-yBopFZjjMF).
4. Trained Models: FP32 fine-tuned DistilBERT and INT8 dynamic quantized model.
5. FastAPI Service: Production-ready API with schema validation and confidence-based routing gates.
6. Benchmarking Scripts: Tooling to evaluate quantization efficiency and HTTP latency SLA adherence.
