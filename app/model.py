import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Any


class BankingClassifier:
    """Wrapper class managing model lifecycle, tokenization, and inference."""

    def __init__(self, weights_path: str = "weights", threshold: float = 0.85):
        self.threshold = threshold
        self.weights_path = weights_path
        self.quantized_path = os.path.join("weights_quantized", "model_quantized.pt")

        print("📦 Initializing Banking Classifier...")

        # 1. Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.weights_path)

        # 2. Prefer Quantized INT8 model on CPU if available
        if os.path.exists(self.quantized_path):
            print(f"⚡ Loading Quantized INT8 model from '{self.quantized_path}' on CPU...")
            self.device = torch.device("cpu")
            self.model = torch.load(self.quantized_path, map_location=self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"📦 Loading FP32 model from '{self.weights_path}' on device: {self.device}")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.weights_path)
            self.model.to(self.device)

        self.model.eval()
        self.id2label = self.model.config.id2label

        # 3. Warm-up
        self._warmup()
        print("✅ Model engine ready for inference!")

    def _warmup(self):
        inputs = self.tokenizer(
            "Warmup ticket",
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
            return_token_type_ids=False
        ).to(self.device)
        with torch.inference_mode():
            _ = self.model(**inputs)

    def predict(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
            return_token_type_ids=False
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

        top_idx = int(probabilities.argmax())
        confidence = float(probabilities[top_idx])
        predicted_intent = self.id2label[top_idx]

        escalate = confidence < self.threshold
        route_decision = "HUMAN_ESCALATION" if escalate else "AUTO_ROUTE"

        all_scores = {
            self.id2label[i]: round(float(prob), 4)
            for i, prob in enumerate(probabilities)
        }

        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "route": route_decision,
            "escalate_to_human": escalate,
            "all_scores": all_scores
        }