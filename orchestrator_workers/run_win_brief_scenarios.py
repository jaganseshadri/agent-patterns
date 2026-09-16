"""Run the Deal Win Brief orchestrator against scenarios from a JSON file.

Scenarios live in win_brief_scenarios.json as a list of
{"id", "name", "task", "context"} objects. Pick what to run by editing
SCENARIO below:

    SCENARIO = "ALL"                 -> run every scenario, in file order
    SCENARIO = "zylo_pricing"        -> run only the scenario with that id
    SCENARIO = 0                     -> run only the scenario at that index

Each run reuses process()'s own verbose printing (orchestrator analysis,
per-worker output, synthesis) — this script just selects which scenario(s)
feed into it and runs them one after another.
"""

import json
import sys

from flexible_orchestrator_worker_pattern import FlexibleOrchestrator
from win_brief_prompts import ORCHESTRATOR_PROMPT, SYNTHESIS_PROMPT, WORKER_PROMPT

SCENARIOS_FILE = "win_brief_scenarios.json"

# Config: "ALL", a scenario id (str), or a scenario index (int).
SCENARIO = "ALL"


def load_scenarios(path: str = SCENARIOS_FILE) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def select_scenarios(scenarios: list[dict], selector) -> list[dict]:
    if selector == "ALL":
        return scenarios

    if isinstance(selector, int):
        if not (0 <= selector < len(scenarios)):
            raise IndexError(
                f"Scenario index {selector} out of range (0-{len(scenarios) - 1})"
            )
        return [scenarios[selector]]

    matches = [s for s in scenarios if s["id"] == selector]
    if not matches:
        available = ", ".join(s["id"] for s in scenarios)
        raise ValueError(f"No scenario with id {selector!r}. Available: {available}")
    return matches


def run_scenario(orchestrator: FlexibleOrchestrator, scenario: dict) -> dict:
    print("\n" + "#" * 80)
    print(f"# SCENARIO [{scenario['id']}]: {scenario['name']}")
    print("#" * 80)
    return orchestrator.process(
        task=scenario["task"],
        context=scenario.get("context", {}),
    )


def main() -> dict:
    scenarios = load_scenarios()
    to_run = select_scenarios(scenarios, SCENARIO)

    orchestrator = FlexibleOrchestrator(
        orchestrator_prompt=ORCHESTRATOR_PROMPT,
        worker_prompt=WORKER_PROMPT,
        synthesis_prompt=SYNTHESIS_PROMPT,
    )

    all_results = {}
    for scenario in to_run:
        all_results[scenario["id"]] = run_scenario(orchestrator, scenario)

    return all_results


if __name__ == "__main__":
    # Optional CLI override: python run_win_brief_scenarios.py zylo_pricing
    #                        python run_win_brief_scenarios.py 2
    #                        python run_win_brief_scenarios.py ALL
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        SCENARIO = int(arg) if arg.isdigit() else arg

    main()
