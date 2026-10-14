import time
import requests
import statistics

API_URL = "http://127.0.0.1:8000/predict"

payloads = [
    {"ticket_id": "TCK-1", "text": "My card got stuck in the ATM machine."},
    {"ticket_id": "TCK-2", "text": "I want to reset my internet banking password."},
    {"ticket_id": "TCK-3", "text": "How do I make an international wire transfer?"},
    {"ticket_id": "TCK-4", "text": "There is a fraudulent charge of $400 on my account."},
    {"ticket_id": "TCK-5", "text": "What is the interest rate on the premium savings plan?"}
]

print("Verifying API connectivity...")
try:
    health = requests.get("http://127.0.0.1:8000/health")
    print(f"Server Health: {health.json()}")
except Exception as e:
    print(f"❌ Error: API server is not running on port 8000. Start it first with uvicorn! Details: {e}")
    exit(1)

print("\nRunning benchmark: Sending 100 real HTTP POST requests...")
latencies = []

# Warm-up request
requests.post(API_URL, json=payloads[0])

for i in range(20):
    for payload in payloads:
        start_time = time.perf_counter()
        response = requests.post(API_URL, json=payload)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        if response.status_code == 200:
            latencies.append(latency_ms)
        else:
            print(f"Failed request: {response.status_code}")

avg_latency = statistics.mean(latencies)
p50 = statistics.median(latencies)
p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile

print("\n" + "=" * 50)
print("🚀 API END-TO-END HTTP LATENCY REPORT")
print("=" * 50)
print(f"Total Requests:      {len(latencies)}")
print(f"Average Latency:     {avg_latency:.2f} ms")
print(f"Median (p50):        {p50:.2f} ms")
print(f"95th Percentile:     {p95:.2f} ms")
print(f"Goal Met (<100ms):   {'✅ PASSED' if avg_latency < 100 else '❌ FAILED'}")
print("=" * 50)