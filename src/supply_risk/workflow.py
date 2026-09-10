from __future__ import annotations

import json

from .models import AgentMessage, CaseStatus, DecisionStatus, MitigationRecommendation, WorkflowState
from .tools import ComponentImpactTool, PurchaseOrderRiskTool


def apply_human_checkpoint(state: WorkflowState, approved: bool | None) -> DecisionStatus:
    uncertain = any(signal.confidence < 0.7 or signal.data_quality_issue for signal in state.risk_signals)
    incomplete = not state.impacts
    if uncertain or incomplete:
        state.decision_status = DecisionStatus.ESCALATED
        state.case_status = CaseStatus.ESCALATED
        rationale = "Escalated because evidence is conflicting or inventory impact cannot be calculated."
    elif approved is None:
        state.decision_status = DecisionStatus.NEEDS_HUMAN_REVIEW
        state.case_status = CaseStatus.NEEDS_HUMAN_REVIEW
        rationale = "A planner must approve the mitigation recommendation."
    elif approved:
        state.decision_status = DecisionStatus.RECOMMENDATION_READY
        state.case_status = CaseStatus.APPROVED
        rationale = "Planner approved the proposed mitigation."
    else:
        state.decision_status = DecisionStatus.ESCALATED
        state.case_status = CaseStatus.ESCALATED
        rationale = "Planner rejected the recommendation and requested review."
    state.messages.append(AgentMessage("human_checkpoint", "workflow", "decision", {"status": state.decision_status, "rationale": rationale}))
    return state.decision_status


def default_recommendation(state: WorkflowState) -> None:
    if state.impacts and state.impacts[0].projected_shortage_quantity:
        expedite = next((option for option in state.action_catalogue if option.action_id == "EXPEDITE" and option.eligible), None)
        if expedite is None:
            return
        state.recommendations.append(MitigationRecommendation(expedite.label, 1, expedite.estimated_cost,
            state.recommendation_explanation or "Expediting reduces exposure before the expected delayed receipt.",
            expedite.lead_time_reduction_days, expedite.reversible, True))
        state.approval_reason = f"Approval is required for the {expedite.label.lower()} action at an estimated cost of {expedite.estimated_cost}."
        state.case_status = CaseStatus.RECOMMENDATION_READY
    elif state.impacts:
        monitor = next(option for option in state.action_catalogue if option.action_id == "MONITOR_ONLY" and option.eligible)
        state.recommendations.append(MitigationRecommendation(monitor.label, 1, monitor.estimated_cost,
            state.recommendation_explanation or "Projected inventory remains above safety stock through the expected receipt date.",
            monitor.lead_time_reduction_days, monitor.reversible, False))
        state.approval_reason = "No approval is required because monitoring does not create an operational commitment."
        state.case_status = CaseStatus.RECOMMENDATION_READY


def validate_synthetic_inputs(state: WorkflowState) -> None:
    missing_fields = []
    for position in state.inventory:
        if position.on_hand_quantity is None:
            missing_fields.append(f"inventory.on_hand_quantity:{position.component_id}")
    state.case_status = CaseStatus.VALIDATED
    state.messages.append(AgentMessage("validator", "workflow", "validation", {"missing_fields": missing_fields}))


def escalate(state: WorkflowState, reason: str) -> DecisionStatus:
    state.decision_status = DecisionStatus.ESCALATED
    state.case_status = CaseStatus.ESCALATED
    state.messages.append(AgentMessage("policy_gate", "workflow", "decision", {"status": state.decision_status, "rationale": reason}))
    return state.decision_status


def run_poc_workflow(state: WorkflowState, approved: bool | None) -> DecisionStatus:
    validate_synthetic_inputs(state)
    order = state.purchase_orders[0]

    risk_payload = json.loads(PurchaseOrderRiskTool(state=state)._run(order.purchase_order_id))
    state.case_status = CaseStatus.RISK_ASSESSED
    state.messages.append(AgentMessage("risk_intelligence", "supply_impact", "risk_assessment", risk_payload))
    risk_issue = next((signal for signal in state.risk_signals if signal.confidence < 0.7 or signal.data_quality_issue), None)
    missing_fields = state.messages[0].payload.get("missing_fields", [])
    if risk_issue:
        reasons = [risk_issue.data_quality_issue or "low risk confidence"] + missing_fields
        return escalate(state, "Escalated before impact/mitigation because evidence failed policy gates: " + ", ".join(reasons))

    impact_payload = json.loads(ComponentImpactTool(state=state)._run(order.component_id))
    state.messages.append(AgentMessage("supply_impact", "mitigation_planner", "impact_assessment", impact_payload))
    if impact_payload.get("status") == "insufficient_data" or not state.impacts:
        return escalate(state, "Escalated before mitigation because inventory impact cannot be calculated.")
    state.case_status = CaseStatus.IMPACT_ASSESSED

    default_recommendation(state)
    if not state.recommendations:
        return escalate(state, "No mitigation recommendation was generated for the POC scenario.")
    state.messages.append(AgentMessage("mitigation_planner", "human_checkpoint", "recommendation", {"recommendation": state.recommendations[0].__dict__}))
    if not state.recommendations[0].approval_required:
        state.decision_status = DecisionStatus.RECOMMENDATION_READY
        state.messages.append(AgentMessage("policy_gate", "workflow", "decision", {"status": state.decision_status, "rationale": "Monitor-only recommendation requires no operational action."}))
        return state.decision_status
    return apply_human_checkpoint(state, approved=approved)


def planner_decision_output(state: WorkflowState) -> dict[str, object]:
    risk = state.risk_signals[0] if state.risk_signals else None
    impact = state.impacts[0] if state.impacts else None
    recommendation = state.recommendations[0] if state.recommendations else None
    planner_output = {
        "incident_id": state.incident_id,
        "case_status": state.case_status,
        "risk": None if risk is None else {
            "severity": risk.severity,
            "confidence": risk.confidence,
            "status": "credible" if risk.confidence >= 0.7 and not risk.data_quality_issue else "insufficient_evidence",
            "data_quality_issue": risk.data_quality_issue,
        },
        "impact": None if impact is None else {
            "shortage_quantity": impact.projected_shortage_quantity,
            "shortage_date": impact.shortage_date,
            "production_exposure_units": impact.affected_production_units,
        },
        "recommendation": None if recommendation is None else {
            "action": recommendation.action,
            "estimated_cost": recommendation.estimated_cost,
            "lead_time_reduction_days": recommendation.lead_time_reduction_days,
            "rationale": recommendation.rationale,
            "approval_required": recommendation.approval_required,
        },
        "evidence_summary": state.evidence_summary,
        "impact_summary": state.impact_summary,
        "recommendation_explanation": state.recommendation_explanation,
        "approval_reason": state.approval_reason,
        "decision_status": state.decision_status,
    }
    if impact is None and state.decision_status == DecisionStatus.ESCALATED:
        return {
            **planner_output,
            "impact": {
                "status": "insufficient_data",
                "shortage_quantity": None,
                "shortage_date": None,
                "production_exposure_units": None,
            },
        }
    return planner_output