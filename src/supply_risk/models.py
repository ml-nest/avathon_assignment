from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionStatus(StrEnum):
    RECOMMENDATION_READY = "recommendation_ready"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    ESCALATED = "escalated"


class CaseStatus(StrEnum):
    RECEIVED = "received"
    VALIDATED = "validated"
    RISK_ASSESSED = "risk_assessed"
    IMPACT_ASSESSED = "impact_assessed"
    RECOMMENDATION_READY = "recommendation_ready"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    APPROVED = "approved"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class Supplier:
    supplier_id: str
    name: str
    reliability_score: float
    normal_lead_time_days: int
    expedite_available: bool
    expedite_cost: int | None
    alternate_supplier_id: str | None


@dataclass(frozen=True)
class PurchaseOrder:
    purchase_order_id: str
    supplier_id: str
    component_id: str
    quantity: int
    promised_receipt_date: date
    shipment_status: str
    reported_delay_days: int | None
    supplier_reported_eta: date | None = None
    transport_eta: date | None = None
    transport_milestone: str | None = None


@dataclass(frozen=True)
class InventoryPosition:
    component_id: str
    on_hand_quantity: int | None
    safety_stock: int
    daily_consumption: int


@dataclass(frozen=True)
class RiskSignal:
    purchase_order_id: str
    severity: Severity
    confidence: float
    rationale: str
    data_quality_issue: str | None = None


@dataclass(frozen=True)
class ImpactAssessment:
    component_id: str
    projected_shortage_quantity: int
    shortage_date: date | None
    affected_production_units: int
    severity: Severity
    rationale: str


@dataclass(frozen=True)
class MitigationRecommendation:
    action: str
    priority: int
    estimated_cost: int
    rationale: str
    lead_time_reduction_days: int = 0
    reversible: bool = True
    approval_required: bool = True


@dataclass(frozen=True)
class ActionOption:
    action_id: str
    label: str
    estimated_cost: int
    lead_time_reduction_days: int
    reversible: bool
    eligible: bool


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    message_type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=str))


@dataclass
class WorkflowState:
    as_of_date: date
    suppliers: list[Supplier]
    purchase_orders: list[PurchaseOrder]
    inventory: list[InventoryPosition]
    action_catalogue: list[ActionOption] = field(default_factory=list)
    incident_id: str = "INC-001"
    messages: list[AgentMessage] = field(default_factory=list)
    risk_signals: list[RiskSignal] = field(default_factory=list)
    impacts: list[ImpactAssessment] = field(default_factory=list)
    recommendations: list[MitigationRecommendation] = field(default_factory=list)
    evidence_summary: str | None = None
    impact_summary: str | None = None
    recommendation_explanation: str | None = None
    approval_reason: str | None = None
    decision_status: DecisionStatus | None = None
    case_status: CaseStatus = CaseStatus.RECEIVED
