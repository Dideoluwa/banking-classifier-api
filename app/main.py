import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, status
from app.schemas import TicketRequest, TicketResponse
from app.model import BankingClassifier
from app.database import (
    init_db,
    record_prediction,
    fetch_recent_logs,
    fetch_system_stats,
    active_dialect
)

# Global reference for the classifier
classifier: BankingClassifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown.
    Loads the model once when the web server boots and initializes DB schema.
    """
    global classifier
    print("🚀 Booting Banking Classifier Service...")
    # Initialize database tables
    init_db()
    # Initialize classification model
    classifier = BankingClassifier(weights_path="weights", threshold=0.85)
    yield
    print("🛑 Shutting down Banking Classifier Service...")


# Initialize FastAPI application
app = FastAPI(
    title="Banking Ticket Triage & Intent Classifier API",
    description="Automated routing and human escalation microservice powered by fine-tuned DistilBERT and PostgreSQL.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
async def health_check():
    """
    Health check endpoint for container orchestrators, load balancers, and monitoring.
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model engine not loaded"
        )
    return {
        "status": "healthy",
        "device": str(classifier.device),
        "threshold": classifier.threshold,
        "database": active_dialect
    }


@app.post("/predict", response_model=TicketResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
async def predict_ticket(payload: TicketRequest, background_tasks: BackgroundTasks):
    """
    Accepts customer ticket text, classifies intent, applies confidence threshold routing,
    and asynchronously logs the request and result to the database.
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready to accept requests"
        )

    try:
        start_time = time.perf_counter()
        # Run inference through the model engine
        result = classifier.predict(payload.text)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Asynchronously log prediction to the database without delaying client response
        background_tasks.add_task(
            record_prediction,
            ticket_id=payload.ticket_id,
            text=payload.text,
            predicted_intent=result["intent"],
            confidence=result["confidence"],
            route=result["route"],
            escalate_to_human=result["escalate_to_human"],
            latency_ms=latency_ms
        )

        # Structure response according to TicketResponse schema
        return TicketResponse(
            ticket_id=payload.ticket_id,
            intent=result["intent"],
            confidence=result["confidence"],
            route=result["route"],
            escalate_to_human=result["escalate_to_human"],
            all_scores=result["all_scores"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failure: {str(e)}"
        )


@app.get("/logs", status_code=status.HTTP_200_OK, tags=["Audit & Monitoring"])
async def get_logs(limit: int = Query(default=50, ge=1, le=500)):
    """
    Retrieves recent prediction records from the database.
    """
    logs = fetch_recent_logs(limit=limit)
    return {
        "count": len(logs),
        "limit": limit,
        "database": active_dialect,
        "logs": logs
    }


@app.get("/stats", status_code=status.HTTP_200_OK, tags=["Audit & Monitoring"])
async def get_stats():
    """
    Calculates operational statistics and throughput across all logged predictions.
    """
    return fetch_system_stats()