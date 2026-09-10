from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from .models import ActionOption, InventoryPosition, PurchaseOrder, Supplier, WorkflowState


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _rows(filename: str) -> list[dict[str, str]]:
    with (DATA_DIR / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _optional_int(value: str) -> int | None:
    return int(value) if value else None


def _optional_date(value: str) -> date | None:
    return date.fromisoformat(value) if value else None


def _bool(value: str) -> bool:
    return value.lower() == "true"


def _has_conflicting_eta(order: PurchaseOrder) -> bool:
    return (
        order.supplier_reported_eta is not None
        and order.transport_eta is not None
        and order.supplier_reported_eta > order.transport_eta
        and order.transport_milestone in {"on_schedule", "delivered"}
    )


def _is_ambiguous_case(order: PurchaseOrder, inventory: InventoryPosition) -> bool:
    return _has_conflicting_eta(order) or inventory.on_hand_quantity is None


def generate_scenario(seed: int = 42, ambiguous: bool = False) -> WorkflowState:
    """Create a reproducible supplier-delay scenario selected by policy logic."""
    _ = seed
    as_of_date = date(2026, 9, 4)
    suppliers = [
        Supplier(
            row["supplier_id"],
            row["name"],
            float(row["reliability_score"]),
            int(row["normal_lead_time_days"]),
            _bool(row["expedite_available"]),
            _optional_int(row["expedite_cost"]),
            row["alternate_supplier_id"] or None,
        )
        for row in _rows("supplier_info.csv")
    ]
    inventory_by_component = {row["component_id"]: row for row in _rows("inventory_info.csv")}
    selected_order: PurchaseOrder | None = None
    selected_inventory: InventoryPosition | None = None
    for row in _rows("purchase_orders.csv"):
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
        inventory = InventoryPosition(
            inventory_row["component_id"],
            _optional_int(inventory_row["on_hand_quantity"]),
            int(inventory_row["safety_stock"]),
            int(inventory_row["daily_consumption"]),
        )
        if _is_ambiguous_case(order, inventory) == ambiguous:
            selected_order = order
            selected_inventory = inventory
            break

    if selected_order is None or selected_inventory is None:
        requested = "ambiguous" if ambiguous else "successful"
        raise ValueError(f"No {requested} supplier-delay case could be identified from fixture logic.")

    supplier = next(item for item in suppliers if item.supplier_id == selected_order.supplier_id)
    action_catalogue = [
        ActionOption(
            "EXPEDITE",
            f"Expedite remaining {selected_order.purchase_order_id} quantity",
            supplier.expedite_cost or 0,
            4,
            True,
            supplier.expedite_available and supplier.expedite_cost is not None,
        ),
        ActionOption("MONITOR_ONLY", "Monitor only", 0, 0, True, True),
    ]
    return WorkflowState(as_of_date, suppliers, [selected_order], [selected_inventory], action_catalogue)
