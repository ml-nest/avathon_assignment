from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from supply_risk.models import ActionOption, InventoryPosition, PurchaseOrder, Supplier, WorkflowState
from supply_risk.crew import build_crew, capture_agent_summaries, finalize_poc_state, planner_decision_output, validate_synthetic_inputs
from supply_risk.synthetic_data import DATA_DIR


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _optional_date(value: str):
    from datetime import date

    return date.fromisoformat(value) if value else None


def _optional_int(value: str) -> int | None:
    return int(value) if value else None


def _is_ambiguous_record(order: PurchaseOrder, inventory: InventoryPosition) -> bool:
    has_conflict = (
        order.supplier_reported_eta is not None
        and order.transport_eta is not None
        and order.supplier_reported_eta > order.transport_eta
        and order.transport_milestone in {"on_schedule", "delivered"}
    )
    return has_conflict or inventory.on_hand_quantity is None


def _record_states() -> list[WorkflowState]:
    from datetime import date

    suppliers = [
        Supplier(row["supplier_id"], row["name"], float(row["reliability_score"]), int(row["normal_lead_time_days"]), row["expedite_available"].lower() == "true", _optional_int(row["expedite_cost"]), row["alternate_supplier_id"] or None)
        for row in _load_rows(DATA_DIR / "supplier_info.csv")
    ]
    inventory_by_component = {row["component_id"]: row for row in _load_rows(DATA_DIR / "inventory_info.csv")}
    states = []

    for row in _load_rows(DATA_DIR / "purchase_orders.csv"):
        order = PurchaseOrder(
            row["purchase_order_id"],
            row["supplier_id"],
            row["component_id"],
            int(row["quantity"]),
            date.fromisoformat(row["promised_receipt_date"]),
            row["shipment_status"],
            _optional_int(row["reported_delay_days"]),
            _optional_date(row["supplier_reported_eta"]),
            _optional_date(row["transport_eta"]),
            row["transport_milestone"] or None,
        )
        inventory_row = inventory_by_component[order.component_id]
        inventory = InventoryPosition(inventory_row["component_id"], _optional_int(inventory_row["on_hand_quantity"]), int(inventory_row["safety_stock"]), int(inventory_row["daily_consumption"]))
        supplier = next(item for item in suppliers if item.supplier_id == order.supplier_id)
        action_catalogue = [
            ActionOption("EXPEDITE", f"Expedite remaining {order.purchase_order_id} quantity", supplier.expedite_cost or 0, 4, True, supplier.expedite_available and supplier.expedite_cost is not None),
            ActionOption("MONITOR_ONLY", "Monitor only", 0, 0, True, True),
        ]
        states.append(WorkflowState(date(2026, 9, 4), suppliers, [order], [inventory], action_catalogue, incident_id=f"INC-{order.purchase_order_id}"))
    return states


def _scan_all_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    traces = []
    for state in _record_states():
        ambiguous = _is_ambiguous_record(state.purchase_orders[0], state.inventory[0])
        validate_synthetic_inputs(state)
        crew = build_crew(state)
        try:
            crew.kickoff()
            capture_agent_summaries(state, crew)
        except Exception as exc:
            state.evidence_summary = f"Agent review failed: {type(exc).__name__}."
            state.impact_summary = "Impact summary unavailable because the agent workflow did not complete."
            state.recommendation_explanation = "Recommendation summary unavailable because the agent workflow did not complete."
        finalize_poc_state(state, approved=None)
        output = planner_decision_output(state)
        risk = output.get("risk") or {}
        impact = output.get("impact") or {}
        recommendation = output.get("recommendation") or {}
        recommendation_decision = (
            "Approve / Reject"
            if recommendation.get("approval_required")
            else "Not required"
            if recommendation
            else "Escalate"
        )
        rows.append({
            "purchase_order_id": state.purchase_orders[0].purchase_order_id,
            "component_id": state.purchase_orders[0].component_id,
            "supplier_id": state.purchase_orders[0].supplier_id,
            "logic_status": "ambiguous" if ambiguous else "actionable",
            "risk_confidence": risk.get("confidence"),
            "data_quality_issue": risk.get("data_quality_issue"),
            "shortage_quantity": impact.get("shortage_quantity"),
            "shortage_date": impact.get("shortage_date"),
            "recommendation": recommendation.get("action"),
            "decision_status": output.get("decision_status"),
            "Risk Intelligence Summary": output.get("evidence_summary") or "Risk summary unavailable.",
            "Supply Impact Summary": output.get("impact_summary") or "Impact summary unavailable.",
            "Mitigation Planner Summary": output.get("recommendation_explanation") or "Mitigation summary unavailable.",
            "Approve / Reject Recommendation": recommendation_decision,
            "approval_reason": output.get("approval_reason"),
        })
        for message in state.messages:
            trace = message.to_dict()
            trace["purchase_order_id"] = state.purchase_orders[0].purchase_order_id
            traces.append(trace)
    return rows, traces


st.set_page_config(page_title="Supply Chain Risk Copilot", layout="wide")
st.markdown(
    """
    <style>
    .roster-header { background: #102a43; color: #f7fafc; padding: 0.55rem 0.8rem; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .roster-kicker { color: #2f855a; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Supply Chain Risk Copilot")

st.caption("Live supplier operations roster")

st.subheader("Synthetic Operational Data")
tabs = st.tabs(["Purchase Orders", "Inventory", "Suppliers"])
for tab, filename in zip(tabs, ["purchase_orders.csv", "inventory_info.csv", "supplier_info.csv"]):
    with tab:
        st.dataframe(_load_rows(DATA_DIR / filename), width="stretch", hide_index=True)

st.divider()
st.subheader("Logic-Derived Workflow Outcomes")
st.markdown(
    '<div class="roster-header">Incident &nbsp;&nbsp; Component &nbsp;&nbsp; Supplier &nbsp;&nbsp; Issue summary &nbsp;&nbsp; Decision</div>',
    unsafe_allow_html=True,
)
refresh_col, status_col = st.columns([1, 3])
with refresh_col:
    refresh = st.button("Refresh tables", type="primary")
if refresh or "roster_rows" not in st.session_state:
    with st.spinner("Evaluating purchase orders..."):
        roster_rows, trace_rows = _scan_all_records()
    st.session_state["roster_rows"] = roster_rows
    st.session_state["trace_rows"] = trace_rows
    st.session_state["roster_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
with status_col:
    st.caption(f"Last refresh: {st.session_state.get('roster_updated', 'not yet run')}")

roster_rows = st.session_state["roster_rows"]
counts = {
    "Needs review": sum(row["decision_status"] == "needs_human_review" for row in roster_rows),
    "Escalated": sum(row["decision_status"] == "escalated" for row in roster_rows),
    "Recommendation ready": sum(row["decision_status"] == "recommendation_ready" for row in roster_rows),
}
metric_cols = st.columns(3)
for column, (label, value) in zip(metric_cols, counts.items()):
    column.metric(label, value)

st.dataframe(roster_rows, width="stretch", hide_index=True)

with st.expander("Workflow Trace"):
    st.dataframe(st.session_state.get("trace_rows", []), width="stretch", hide_index=True)