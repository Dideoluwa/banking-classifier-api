import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from app.db_models import Base, PredictionLog

DEFAULT_PG_URL = f"postgresql://{os.getenv('USER', 'don')}@localhost:5432/banking_db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_PG_URL)

engine = None
SessionLocal = None
active_dialect = "unknown"

def get_engine():
    global engine, SessionLocal, active_dialect
    if engine is not None:
        return engine

    try:
        # Try connecting with the configured PostgreSQL URL
        test_engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3} if "postgresql" in DATABASE_URL else {}
        )
        with test_engine.connect() as conn:
            pass  # Connectivity verified
        engine = test_engine
        active_dialect = "PostgreSQL"
        print(f"Connected to database ({active_dialect}): {DATABASE_URL.split('@')[-1]}")
    except Exception as e:
        print(f"PostgreSQL connection failed ({e}). Falling back to local SQLite...")
        fallback_url = "sqlite:///./predictions.db"
        engine = create_engine(fallback_url, connect_args={"check_same_thread": False})
        active_dialect = "SQLite (fallback)"
        print(f"Connected to database ({active_dialect}): predictions.db")

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine


def init_db():
    """Initializes tables on startup."""
    eng = get_engine()
    Base.metadata.create_all(bind=eng)
    print(f"Database schema initialized on {active_dialect}.")


def record_prediction(
    ticket_id: str,
    text: str,
    predicted_intent: str,
    confidence: float,
    route: str,
    escalate_to_human: bool,
    latency_ms: float
):
    """
    Background worker function that persists inference logs to the database.
    Runs asynchronously without delaying HTTP responses.
    """
    if SessionLocal is None:
        get_engine()

    session = SessionLocal()
    try:
        log_entry = PredictionLog(
            ticket_id=ticket_id,
            text=text,
            predicted_intent=predicted_intent,
            confidence=confidence,
            route=route,
            escalate_to_human=escalate_to_human,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow()
        )
        session.add(log_entry)
        session.commit()
    except Exception as err:
        session.rollback()
        print(f"Failed to persist prediction log: {err}", file=sys.stderr)
    finally:
        session.close()


def fetch_recent_logs(limit: int = 50):
    """Retrieves recent prediction logs ordered by newest first."""
    if SessionLocal is None:
        get_engine()

    session = SessionLocal()
    try:
        records = session.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit).all()
        return [r.to_dict() for r in records]
    finally:
        session.close()


def fetch_system_stats():
    """Calculates audit metrics across all logged predictions."""
    if SessionLocal is None:
        get_engine()

    session = SessionLocal()
    try:
        total = session.query(func.count(PredictionLog.id)).scalar() or 0
        if total == 0:
            return {
                "total_predictions": 0,
                "database_engine": active_dialect,
                "auto_route_count": 0,
                "human_escalation_count": 0,
                "auto_route_percentage": 0.0,
                "average_latency_ms": 0.0,
                "intent_distribution": {}
            }

        auto_count = session.query(func.count(PredictionLog.id)).filter(PredictionLog.route == "AUTO_ROUTE").scalar() or 0
        escalate_count = session.query(func.count(PredictionLog.id)).filter(PredictionLog.route == "HUMAN_ESCALATION").scalar() or 0
        avg_latency = session.query(func.avg(PredictionLog.latency_ms)).scalar() or 0.0

        intents = session.query(PredictionLog.predicted_intent, func.count(PredictionLog.id)).group_by(PredictionLog.predicted_intent).all()
        intent_dist = {intent: count for intent, count in intents}

        return {
            "total_predictions": total,
            "database_engine": active_dialect,
            "auto_route_count": auto_count,
            "human_escalation_count": escalate_count,
            "auto_route_percentage": round((auto_count / total) * 100, 2),
            "average_latency_ms": round(float(avg_latency), 2),
            "intent_distribution": intent_dist
        }
    finally:
        session.close()
