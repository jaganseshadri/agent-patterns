"""One-off mock verification -- not a permanent test file.

Patches llm_call so nothing hits the real OpenRouter API, and exercises:
  1. Full process() flow: decompose -> workers -> synthesis
  2. Retry-then-succeed behavior
  3. MissingAPIKeyError fails fast with zero retries
  4. Scenario selection logic (ALL / id / index / invalid)
  5. parse_tasks() XML parsing directly
"""

import time
from unittest.mock import patch

import flexible_orchestrator_worker_pattern as fowp
import run_win_brief_scenarios as runner
from util import MissingAPIKeyError
from win_brief_prompts import ORCHESTRATOR_PROMPT, WORKER_PROMPT

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(label)


ORCH_RESPONSE = """
<analysis>
Fake analysis: this deal needs a pricing angle and a security angle.
</analysis>
<tasks>
    <task>
    <type>pricing_roi</type>
    <description>Address the pricing objection directly</description>
    </task>
    <task>
    <type>security_compliance</type>
    <description>Address the security concern directly</description>
    </task>
</tasks>
"""

SYNTHESIS_RESPONSE = """
<response>
Fake synthesized win brief combining both angles.
</response>
"""


def fake_llm_call(prompt, system_prompt="", model=None, max_tokens=4096):
    if "<tasks>" in prompt and "<analysis>" in prompt:
        return ORCH_RESPONSE
    if "Worker results:" in prompt or "Section briefings:" in prompt:
        return SYNTHESIS_RESPONSE
    # worker prompt -- echo back which task_type it was asked to handle
    marker = "pricing_roi" if "pricing_roi" in prompt else "security_compliance"
    return f"<response>Fake worker output for {marker}.</response>"


print("=" * 70)
print("1) FULL PROCESS() FLOW (decompose -> workers -> synthesis)")
print("=" * 70)
with patch.object(fowp, "llm_call", side_effect=fake_llm_call):
    orchestrator = fowp.FlexibleOrchestrator(
        orchestrator_prompt=ORCHESTRATOR_PROMPT,  # real template -- contains <analysis>/<tasks>
        worker_prompt=WORKER_PROMPT,
    )
    result = orchestrator.process(task="Test deal", context={})

check("analysis captured", "Fake analysis" in result["analysis"])
check("two worker results returned", len(result["worker_results"]) == 2, str(result["worker_results"]))
check(
    "worker results match their assigned type",
    all(r["type"] in r["result"] for r in result["worker_results"]),
)
check("synthesis captured", "Fake synthesized win brief" in result["synthesis"])

print("\n" + "=" * 70)
print("2) RETRY-THEN-SUCCEED")
print("=" * 70)
call_count = {"n": 0}


def flaky_then_ok(prompt, system_prompt="", model=None, max_tokens=4096):
    call_count["n"] += 1
    if call_count["n"] == 1:
        raise ConnectionError("simulated transient failure")
    return "<response>recovered</response>"


with patch.object(fowp, "llm_call", side_effect=flaky_then_ok), patch.object(fowp.time, "sleep", lambda s: None):
    orch = fowp.FlexibleOrchestrator(orchestrator_prompt="x", worker_prompt="x", max_retries=2)
    output = orch._call_with_retry("prompt", "some-model", "TestLabel")

check("retry recovered after one failure", output == "<response>recovered</response>", output)
check("exactly 2 calls made (1 fail + 1 success)", call_count["n"] == 2, call_count["n"])

print("\n" + "=" * 70)
print("3) MISSING API KEY FAILS FAST (no retry loop)")
print("=" * 70)
raise_count = {"n": 0}


def always_missing_key(prompt, system_prompt="", model=None, max_tokens=4096):
    raise_count["n"] += 1
    raise MissingAPIKeyError("no key")


with patch.object(fowp, "llm_call", side_effect=always_missing_key), patch.object(fowp.time, "sleep", lambda s: None):
    orch = fowp.FlexibleOrchestrator(orchestrator_prompt="x", worker_prompt="x", max_retries=2)
    raised = False
    try:
        orch._call_with_retry("prompt", "some-model", "TestLabel")
    except MissingAPIKeyError:
        raised = True

check("MissingAPIKeyError propagated (not swallowed)", raised)
check("exactly 1 call made (no retries on config error)", raise_count["n"] == 1, raise_count["n"])

print("\n" + "=" * 70)
print("4) SCENARIO SELECTION LOGIC")
print("=" * 70)
scenarios = runner.load_scenarios()
check("4 scenarios loaded", len(scenarios) == 4, len(scenarios))

all_selected = runner.select_scenarios(scenarios, "ALL")
check("'ALL' returns all 4", len(all_selected) == 4)

by_id = runner.select_scenarios(scenarios, "stalling_close")
check("id selection returns 1 correct scenario", len(by_id) == 1 and by_id[0]["id"] == "stalling_close")

by_index = runner.select_scenarios(scenarios, 2)
check(
    "index selection returns correct scenario",
    by_index[0]["id"] == scenarios[2]["id"],
    by_index[0]["id"],
)

bad_id_raised = False
try:
    runner.select_scenarios(scenarios, "not_a_real_id")
except ValueError:
    bad_id_raised = True
check("unknown id raises ValueError", bad_id_raised)

bad_index_raised = False
try:
    runner.select_scenarios(scenarios, 99)
except IndexError:
    bad_index_raised = True
check("out-of-range index raises IndexError", bad_index_raised)

print("\n" + "=" * 70)
print("5) parse_tasks() XML PARSING")
print("=" * 70)
parsed = fowp.parse_tasks(ORCH_RESPONSE)
check("parsed 2 tasks", len(parsed) == 2, parsed)
check(
    "task fields correct",
    parsed[0]["type"] == "pricing_roi" and "pricing objection" in parsed[0]["description"],
    parsed,
)

print("\n" + "=" * 70)
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 70)
