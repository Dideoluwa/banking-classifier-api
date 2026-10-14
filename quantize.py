import os
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Paths setup
WEIGHTS_PATH = "weights"
QUANTIZED_DIR = "weights_quantized"
os.makedirs(QUANTIZED_DIR, exist_ok=True)


print("Step 1: Loading original FP32 model onto CPU...")
tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_PATH)
model_fp32 = AutoModelForSequenceClassification.from_pretrained(WEIGHTS_PATH)
model_fp32.to("cpu")
model_fp32.eval()


print("Step 2: Applying Dynamic INT8 Quantization to linear layers...")
model_int8 = torch.ao.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)

print("Step 3: Saving quantized artifact...")
quantized_model_file = os.path.join(QUANTIZED_DIR, "model_quantized.pt")
torch.save(model_int8, quantized_model_file)
tokenizer.save_pretrained(QUANTIZED_DIR)

# Compare file sizes on disk
safetensors_path = os.path.join(WEIGHTS_PATH, "model.safetensors")
bin_path = os.path.join(WEIGHTS_PATH, "pytorch_model.bin")
orig_file = safetensors_path if os.path.exists(safetensors_path) else bin_path

fp32_size_mb = os.path.getsize(orig_file) / (1024 * 1024)
int8_size_mb = os.path.getsize(quantized_model_file) / (1024 * 1024)
size_reduction = (1 - (int8_size_mb / fp32_size_mb)) * 100

print("\n" + "=" * 50)
print("📦 STORAGE COMPARISON")
print("=" * 50)
print(f"Original FP32 model:  {fp32_size_mb:.2f} MB")
print(f"Quantized INT8 model: {int8_size_mb:.2f} MB")
print(f"Disk reduction:       {size_reduction:.1f}%")
print("=" * 50)

def tokenize_text(text: str):
    return tokenizer(
        text,
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding=True,
        return_token_type_ids=False  
    )

print("\nStep 4: Benchmarking inference latency on CPU...")

test_queries = [
    "My credit card was charged twice for an ATM withdrawal yesterday.",
    "I forgot my password and cannot sign into the mobile app.",
    "How long does it take for an international wire transfer to arrive?",
    "Someone opened an account with my details, this is fraud.",
    "What are your interest rates on fixed savings accounts?"
]

# Warm-up passes
for query in test_queries:
    inputs = tokenize_text(query)
    with torch.inference_mode():
        _ = model_fp32(**inputs)
        _ = model_int8(**inputs)

# Benchmark FP32 model (100 total runs)
start_fp32 = time.perf_counter()
for _ in range(20):
    for query in test_queries:
        inputs = tokenize_text(query)
        with torch.inference_mode():
            _ = model_fp32(**inputs)
fp32_avg_latency_ms = ((time.perf_counter() - start_fp32) / 100) * 1000

# Benchmark INT8 model (100 total runs)
start_int8 = time.perf_counter()
for _ in range(20):
    for query in test_queries:
        inputs = tokenize_text(query)
        with torch.inference_mode():
            _ = model_int8(**inputs)
int8_avg_latency_ms = ((time.perf_counter() - start_int8) / 100) * 1000

speedup_factor = fp32_avg_latency_ms / int8_avg_latency_ms

print("\n" + "=" * 50)
print("⚡ LATENCY BENCHMARK (Average per single request)")
print("=" * 50)
print(f"Original FP32 Latency:  {fp32_avg_latency_ms:.2f} ms")
print(f"Quantized INT8 Latency: {int8_avg_latency_ms:.2f} ms")
print(f"Speedup Factor:         {speedup_factor:.2f}x faster")
print("=" * 50)


print("\nStep 5: Verifying classification agreement...")
id2label = model_fp32.config.id2label
agreements = 0

for query in test_queries:
    inputs = tokenize_text(query)
    with torch.inference_mode():
        pred_fp32 = int(model_fp32(**inputs).logits.argmax().item())
        pred_int8 = int(model_int8(**inputs).logits.argmax().item())

    match = pred_fp32 == pred_int8
    if match:
        agreements += 1

    print(f"Query: \"{query[:40]}...\"")
    print(f"  FP32: {id2label[pred_fp32]} | INT8: {id2label[pred_int8]} -> {'✅ MATCH' if match else '❌ MISMATCH'}")

agreement_pct = (agreements / len(test_queries)) * 100
print(f"\nFinal Agreement Rate: {agreements}/{len(test_queries)} ({agreement_pct:.0f}%)")