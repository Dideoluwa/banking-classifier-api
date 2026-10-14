from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PredictionLog(Base):
    """
    Table storing inference logs for auditing, monitoring, and active learning.
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ticket_id = Column(String(64), nullable=True, index=True)
    text = Column(Text, nullable=False)
    predicted_intent = Column(String(64), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    route = Column(String(32), nullable=False, index=True)
    escalate_to_human = Column(Boolean, nullable=False)
    latency_ms = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "ticket_id": self.ticket_id,
            "text": self.text,
            "predicted_intent": self.predicted_intent,
            "confidence": round(self.confidence, 4),
            "route": self.route,
            "escalate_to_human": self.escalate_to_human,
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
