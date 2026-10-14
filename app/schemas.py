from pydantic import BaseModel, Field
from typing import Optional, Dict


class TicketRequest(BaseModel):
    """Schema for incoming ticket payload."""
    text: str = Field(
        ...,
        min_length=3,
        description="The customer support message or inquiry text.",
        json_schema_extra={"example": "My card was charged twice for a transfer yesterday."}
    )
    ticket_id: Optional[str] = Field(
        default=None,
        description="Optional tracking ID for the ticket.",
        json_schema_extra={"example": "TCK-8921"}
    )


class TicketResponse(BaseModel):
    """Schema for model prediction and routing decision."""
    ticket_id: Optional[str] = Field(
        default=None,
        description="Ticket ID if provided in the request."
    )
    intent: str = Field(
        ...,
        description="Predicted banking department class."
    )
    confidence: float = Field(
        ...,
        description="Softmax probability score of the top predicted class (0.0 to 1.0)."
    )
    route: str = Field(
        ...,
        description="'AUTO_ROUTE' if confidence meets the threshold, otherwise 'HUMAN_ESCALATION'."
    )
    escalate_to_human: bool = Field(
        ...,
        description="True if confidence is below 0.85, alerting human agents to intervene."
    )
    all_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Softmax probability breakdown across all 5 departments."
    )