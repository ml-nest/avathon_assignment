from __future__ import annotations

import os
import json

# `python-dotenv` is available through the project dependencies, but this
# fallback keeps the module importable even if someone runs only the core Python
# files without installing optional local-environment support.
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from .crew import build_crew, capture_agent_summaries, finalize_poc_state, planner_decision_output, run_poc_workflow, validate_synthetic_inputs
from .synthetic_data import generate_scenario


# Load `.env` before reading CREWAI_LLM or SCENARIO.
if load_dotenv is not None:
    load_dotenv()


def run(ambiguous: bool = False) -> None:
    """Execute one synthetic POC scenario and print the planner contract.

    `ambiguous=False` runs Scenario A, the successful expedite path.
    `ambiguous=True` runs Scenario B, the conflicting-evidence escalation path.
    """
    # Scenario A simulates planner approval so the demo can show the approved
    # recommendation path. Scenario B never approves because policy gates should
    # escalate before any action recommendation is allowed.
    approved = None if ambiguous else True

    state = generate_scenario(ambiguous=ambiguous)
    if os.getenv("USE_CREWAI", "1") == "1":
        validate_synthetic_inputs(state)
        crew = build_crew(state)
        try:
            crew.kickoff()
            capture_agent_summaries(state, crew)
            print("crew_status: completed")
        except Exception as exc:
            # External LLM failures are reported without fabricating a recommendation.
            print(f"crew_status: failed_closed ({type(exc).__name__})")
        print(finalize_poc_state(state, approved=approved))
    else:
        print("crew_status: deterministic_fallback")
        print(run_poc_workflow(state, approved=approved))

    # Print the trace messages so a reviewer can see validation, tool calls,
    # policy gates, recommendations, and simulated human decisions in order.
    for message in state.messages:
        print(message.to_dict())

    # The final JSON object is the planner-facing decision contract described in
    # the documentation and asserted by the acceptance tests.
    print(json.dumps(planner_decision_output(state), default=str, indent=2))


if __name__ == "__main__":
    # `SCENARIO=ambiguous` selects Scenario B. Any other value, or an unset
    # variable, runs Scenario A.
    run(ambiguous=os.getenv("SCENARIO", "success") == "ambiguous")