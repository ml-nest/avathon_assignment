# Supply Chain Risk Copilot

A CrewAI-based decision-support system for early supplier-delay detection, inventory-impact analysis, and approval-gated mitigation planning.

## Business Problem

Manufacturers often discover supplier delays after component inventory is already at risk of causing a production interruption. Planners reconcile supplier updates, purchase orders, and inventory positions manually, resulting in delayed and inconsistent mitigation decisions.

This copilot identifies credible delay risk, quantifies component shortage exposure, and proposes a POC expedite recommendation from a deterministic synthetic action catalogue. It does not execute procurement actions: the POC uses a simulated human approval or escalation checkpoint.

## Why CrewAI

CrewAI makes the role separation observable: each agent receives a narrowly scoped task, uses only the relevant tools, and passes its result into the next task. A single LLM prompt would obscure these decision boundaries and make it harder to inspect failures.

The sequential crew contains:

- **Risk Intelligence Analyst**: inspects a purchase order using `purchase_order_risk_lookup` and records evidence, confidence, and data-quality concerns.
- **Supply Impact Analyst**: invokes `component_impact_calculator` to calculate projected component shortage and production exposure.
- **Mitigation Planner**: uses prior assessments to produce approval-gated mitigation advice or an escalation decision.

The inter-agent contract is represented by `AgentMessage` and typed domain models in `src/supply_risk/models.py`.

## Synthetic Data

Synthetic operational inputs are stored as CSV fixtures under `data/`. `generate_scenario()` loads these files and intentionally excludes disruption ground truth from agent inputs.

| CSV | Contents |
|---|---|
| `data/supplier_info.csv` | Supplier reliability, lead time, expedite availability/cost, and alternate supplier reference. |
| `data/purchase_orders.csv` | PO, component, quantity, receipt dates, delay, supplier-reported ETA, shipment status, transport ETA, and transport milestone evidence. |
| `data/inventory_info.csv` | On-hand inventory, safety stock, and daily consumption. |

The fixtures contain ten linked purchase-order cases. Two required scenario paths are supported by `generate_scenario()`; the CSV files do not store scenario labels, so it identifies the requested path by evaluating operational facts:

- **Success**: a reported supplier delay and sufficient inventory data produce a synthetic expedite recommendation and simulated approval.
- **Ambiguous**: supplier delayed ETA conflicts with transport ETA/milestone evidence, and on-hand inventory is missing. The system escalates before mitigation planning and recommends no action.

The remaining records cover standalone ETA conflict, standalone missing inventory, no-delay monitoring, delayed-but-buffered inventory, shortage with supplier expedite unavailable, shortage with no alternate supplier, low supplier reliability, and alternate-supplier availability.

## Setup

Use Python 3.12 or newer. Create an isolated environment, then install pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The presentation path runs the full workflow through CrewAI and an external LLM. Set a provider key and select the model before starting the demo:

```powershell
$env:GOOGLE_API_KEY = "your-key"
$env:CREWAI_LLM = "gemini/gemini-3.6-flash"
```

A local `.env` file is also supported. `CREWAI_LLM` defaults to `gemini/gemini-3.6-flash`; set it to another CrewAI-supported provider model when needed.

If the external CrewAI/Gemini call fails because of provider errors or free-tier quota limits, the script reports `crew_status: failed_closed (...)`.

## Run

Launch the Streamlit demo UI:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

The first screen shows all synthetic operational inputs in tabular tabs, followed by a CrewAI workflow outcome table for every purchase order. Run either an actionable or ambiguous record and inspect the planner-facing JSON plus detailed trace.

You can also run the CLI scenarios directly.

Run the successful-path simulation:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m supply_risk.main
```

Run the ambiguous-data escalation path:

```powershell
$env:PYTHONPATH = "src"
$env:SCENARIO = "ambiguous"
.\.venv\Scripts\python.exe -m supply_risk.main
```

## Acceptance Tests

Run the executable POC acceptance tests:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest tests.test_acceptance
```

The tests cover the successful expedite path, ambiguous escalation path, missing-inventory guardrail, sufficient-buffer monitor-only guardrail, CSV-backed fixture loading, and a local CrewAI smoke test that verifies tool invocation through CrewAI agents.

## Human Checkpoint

`run_poc_workflow()` applies risk and impact gates before mitigation planning. It escalates when ETA evidence conflicts, confidence is below 0.70, or inventory impact cannot be calculated. `apply_human_checkpoint()` records the simulated planner decision for valid recommendations.

## Limitations

The data is synthetic and does not represent a validated production supplier-risk model. A production implementation needs authenticated ERP and supplier-data integrations, monitoring for tool failures and drift, and controlled approval/audit storage.