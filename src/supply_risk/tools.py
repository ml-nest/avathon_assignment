from __future__ import annotations

import json
from datetime import timedelta

from pydantic import BaseModel, Field

class BaseTool:
    name: str
    description: str
    args_schema: type[BaseModel]

    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

from .models import AgentMessage, ImpactAssessment, RiskSignal, Severity, WorkflowState


class PurchaseOrderInput(BaseModel):
    purchase_order_id: str = Field(description="Purchase order identifier to inspect.")


class PurchaseOrderRiskTool(BaseTool):
    name: str = "purchase_order_risk_lookup"
    description: str = "Looks up a purchase order and returns evidence-based delay risk."
    args_schema: type[BaseModel] = PurchaseOrderInput
    state: WorkflowState

    def _run(self, purchase_order_id: str) -> str:
        order = next(item for item in self.state.purchase_orders if item.purchase_order_id == purchase_order_id)
        supplier = next(item for item in self.state.suppliers if item.supplier_id == order.supplier_id)
        conflict = (
            order.supplier_reported_eta is not None
            and order.transport_eta is not None
            and order.supplier_reported_eta > order.transport_eta
            and order.transport_milestone in {"on_schedule", "delivered"}
        )
        if conflict:
            signal = RiskSignal(order.purchase_order_id, Severity.HIGH, 0.45,
                                "Supplier-reported delayed ETA conflicts with transport ETA/milestone evidence.", "conflicting ETA evidence")
        elif order.reported_delay_days:
            signal = RiskSignal(order.purchase_order_id, Severity.HIGH, 0.88,
                                f"Supplier reported a {order.reported_delay_days}-day delay with ETA {order.supplier_reported_eta}.")
        else:
            signal = RiskSignal(order.purchase_order_id, Severity.LOW, 0.8, "No delay reported.")
        self.state.risk_signals.append(signal)
        self.state.messages.append(AgentMessage("purchase_order_risk_lookup", "risk_intelligence", "tool_result", {"signal": signal.__dict__, "supplier_reliability": supplier.reliability_score}))
        return json.dumps({"signal": signal.__dict__, "supplier_reliability": supplier.reliability_score}, default=str)


class ComponentImpactInput(BaseModel):
    component_id: str = Field(description="Component identifier to assess.")


class ComponentImpactTool(BaseTool):
    name: str = "component_impact_calculator"
    description: str = "Calculates inventory shortage and production exposure for a component."
    args_schema: type[BaseModel] = ComponentImpactInput
    state: WorkflowState

    def _run(self, component_id: str) -> str:
        position = next(item for item in self.state.inventory if item.component_id == component_id)
        if position.on_hand_quantity is None:
            self.state.messages.append(AgentMessage("component_impact_calculator", "supply_impact", "tool_result", {"status": "insufficient_data", "reason": "on-hand inventory is missing"}))
            return json.dumps({"status": "insufficient_data", "reason": "on-hand inventory is missing"})
        order = next(item for item in self.state.purchase_orders if item.component_id == component_id)
        arrival = order.promised_receipt_date + timedelta(days=order.reported_delay_days or 0)
        days_until_receipt = (arrival - self.state.as_of_date).days
        required = position.daily_consumption * days_until_receipt
        shortage = max(0, required + position.safety_stock - position.on_hand_quantity)
        if shortage:
            days_to_safety_stock_breach = max(0, (position.on_hand_quantity - position.safety_stock) // position.daily_consumption + 1)
            shortage_date = self.state.as_of_date + timedelta(days=days_to_safety_stock_breach)
        else:
            shortage_date = None
        impact = ImpactAssessment(component_id, shortage, shortage_date, shortage, Severity.HIGH if shortage else Severity.LOW,
                                  "Projected calendar-day consumption from the assessment date up to, excluding, receipt date, including safety stock.")
        self.state.impacts.append(impact)
        self.state.messages.append(AgentMessage("component_impact_calculator", "supply_impact", "tool_result", {"impact": impact.__dict__}))
        return json.dumps({"impact": impact.__dict__}, default=str)
