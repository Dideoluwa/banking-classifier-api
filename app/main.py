from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from app.schemas import TicketRequest, TicketResponse
from app.model import BankingClassifier


# Global reference for the classifier
classifier: BankingClassifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown.
    Loads the model once when the web server boots.
    """
    global classifier
    print("🚀 Booting Banking Classifier Service...")
    classifier = BankingClassifier(weights_path="weights", threshold=0.85)
    yield
    # Cleanup logic (if needed on server shutdown)
    print("🛑 Shutting down Banking Classifier Service...")


# Initialize FastAPI application
app = FastAPI(
    title="Banking Ticket Triage & Intent Classifier API",
    description="Automated routing and human escalation microservice powered by fine-tuned DistilBERT.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
async def health_check():
    """
    Health check endpoint for container orchestrators and load balancers.
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model engine not loaded"
        )
    return {
        "status": "healthy",
        "device": str(classifier.device),
        "threshold": classifier.threshold
    }


@app.post("/predict", response_model=TicketResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
async def predict_ticket(payload: TicketRequest):
    """
    Accepts customer ticket text, classifies intent, and applies confidence threshold routing.
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready to accept requests"
        )

    try:
        # Run inference through the model engine
        result = classifier.predict(payload.text)

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