import time

from util import MissingAPIKeyError, extract_xml, llm_call

# Model configuration — orchestrator gets a stronger model for planning and
# synthesis, workers get a cheaper/faster model since each subtask is
# narrower in scope. Any OpenRouter slug works for either.
ORCHESTRATOR_MODEL = "google/gemini-3.8-flash"
WORKER_MODEL = "google/gemini-3.5-flash-lite"

# Default guidance for how many subtasks the orchestrator should propose.
# Pass a different value to process() to experiment with broader or
# narrower decomposition without touching the prompt itself.
DEFAULT_TASK_COUNT_HINT = "2-3"

DEFAULT_SYNTHESIS_PROMPT = """
You are combining results from independent workers into one final answer.

Original task: {original_task}

Worker results:
{worker_results}

Combine these into a single, cohesive final response. Keep what's strongest
from each, resolve any contradictions between them, and cut anything
redundant. Do not just concatenate the results.

Return your response in this format:
<response>
Your synthesized final answer here.
</response>
"""


def parse_tasks(tasks_xml: str) -> list[dict]:
    """Parse XML tasks into a list of task dictionaries."""
    tasks = []
    current_task = {}

    for line in tasks_xml.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("<task>"):
            current_task = {}
        elif line.startswith("<type>"):
            current_task["type"] = line[6:-7].strip()
        elif line.startswith("<description>"):
            current_task["description"] = line[12:-13].strip()
        elif line.startswith("</task>"):
            if "description" in current_task:
                if "type" not in current_task:
                    current_task["type"] = "default"
                tasks.append(current_task)

    return tasks


class FlexibleOrchestrator:
    """Break down tasks, run them across worker LLMs, and synthesize the results."""

    def __init__(
        self,
        orchestrator_prompt: str,
        worker_prompt: str,
        synthesis_prompt: str | None = None,
        orchestrator_model: str = ORCHESTRATOR_MODEL,
        worker_model: str = WORKER_MODEL,
        max_retries: int = 2,
    ):
        """Initialize with prompt templates, model selection, and retry budget."""
        self.orchestrator_prompt = orchestrator_prompt
        self.worker_prompt = worker_prompt
        self.synthesis_prompt = synthesis_prompt or DEFAULT_SYNTHESIS_PROMPT
        self.orchestrator_model = orchestrator_model
        self.worker_model = worker_model
        self.max_retries = max_retries

    def _format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with variables."""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required prompt variable: {e}") from e

    def _call_with_retry(self, prompt: str, model: str, label: str) -> str:
        """Call the model, retrying with backoff on failure. Returns '' if all attempts fail."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):  # first try + retries
            try:
                return llm_call(prompt, model=model)
            except MissingAPIKeyError:
                raise  # configuration error, not transient -- retrying can't help
            except Exception as e:  # noqa: BLE001 - genuinely want to retry on anything
                last_error = e
                if attempt <= self.max_retries:
                    wait = 2 ** (attempt - 1)
                    print(f"⚠️  {label} call failed (attempt {attempt}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
        print(f"❌ {label} call failed after {self.max_retries + 1} attempt(s): {last_error}")
        return ""

    def process(
        self,
        task: str,
        context: dict | None = None,
        task_count_hint: str = DEFAULT_TASK_COUNT_HINT,
    ) -> dict:
        """Process task by decomposing it, running subtasks, then synthesizing.

        Prints the full step-by-step trace as it goes: the orchestrator's
        initial ask -> its decomposition -> each worker's brief -> each
        worker's output -> the orchestrator's final synthesis ask -> the
        synthesized result. Redirect stdout to a file to capture the whole
        run as a log.
        """
        context = context or {}

        # ---- Step 1: Orchestrator's initial ask ----
        orchestrator_input = self._format_prompt(
            self.orchestrator_prompt, task=task, task_count_hint=task_count_hint, **context
        )

        print("\n" + "=" * 80)
        print("STEP 1 - ORCHESTRATOR'S INITIAL ASK  (model: " + self.orchestrator_model + ")")
        print("=" * 80)
        print(f"\n{orchestrator_input.strip()}\n")

        orchestrator_response = self._call_with_retry(orchestrator_input, self.orchestrator_model, "Orchestrator")

        # Parse orchestrator response
        analysis = extract_xml(orchestrator_response, "analysis")
        tasks_xml = extract_xml(orchestrator_response, "tasks")
        tasks = parse_tasks(tasks_xml)

        print("\n" + "=" * 80)
        print("STEP 1 RESULT - ORCHESTRATOR'S DECOMPOSITION")
        print("=" * 80)
        print(f"\n{analysis}\n")
        print(f"IDENTIFIED {len(tasks)} APPROACH(ES):")
        for i, task_info in enumerate(tasks, 1):
            print(f"\n  {i}. {task_info['type'].upper()}")
            print(f"     {task_info['description']}")

        if not tasks:
            print("\n❌ Orchestrator produced no usable subtasks after retries — aborting.")
            return {
                "orchestrator_input": orchestrator_input,
                "analysis": analysis,
                "worker_results": [],
                "synthesis_input": "",
                "synthesis": "",
            }

        # ---- Step 2: Each worker's brief, then its output ----
        worker_results = []
        for i, task_info in enumerate(tasks, 1):
            worker_input = self._format_prompt(
                self.worker_prompt,
                original_task=task,
                task_type=task_info["type"],
                task_description=task_info["description"],
                **context,
            )

            print("\n" + "=" * 80)
            print(f"STEP 2.{i} - WORKER BRIEF: {task_info['type'].upper()}  (model: {self.worker_model})")
            print("=" * 80)
            print(f"\n{worker_input.strip()}\n")

            worker_response = self._call_with_retry(
                worker_input, self.worker_model, f"Worker '{task_info['type']}'"
            )
            worker_content = extract_xml(worker_response, "response")

            # Validate worker response - handle empty outputs after retries are exhausted
            if not worker_content or not worker_content.strip():
                print(f"⚠️  Warning: Worker '{task_info['type']}' returned no content after retries")
                worker_content = f"[Error: Worker '{task_info['type']}' failed to generate content]"

            print("-" * 80)
            print(f"STEP 2.{i} RESULT - WORKER OUTPUT: {task_info['type'].upper()}")
            print("-" * 80)
            print(f"\n{worker_content}\n")

            worker_results.append(
                {
                    "type": task_info["type"],
                    "description": task_info["description"],
                    "prompt": worker_input,
                    "result": worker_content,
                }
            )

        # ---- Step 3: Orchestrator's final synthesis ask, then its output ----
        worker_results_block = "\n\n".join(
            f"[{r['type'].upper()}]\n{r['result']}" for r in worker_results
        )
        synthesis_input = self._format_prompt(
            self.synthesis_prompt, original_task=task, worker_results=worker_results_block, **context
        )

        print("\n" + "=" * 80)
        print("STEP 3 - ORCHESTRATOR'S FINAL SYNTHESIS ASK  (model: " + self.orchestrator_model + ")")
        print("=" * 80)
        print(f"\n{synthesis_input.strip()}\n")

        synthesis_response = self._call_with_retry(synthesis_input, self.orchestrator_model, "Synthesis")
        synthesis = extract_xml(synthesis_response, "response")

        if not synthesis or not synthesis.strip():
            print("⚠️  Warning: Synthesis returned no content after retries")
            synthesis = "[Error: synthesis failed to generate content]"

        print("-" * 80)
        print("STEP 3 RESULT - FINAL SYNTHESIZED WIN BRIEF")
        print("-" * 80)
        print(f"\n{synthesis}\n")

        return {
            "orchestrator_input": orchestrator_input,
            "analysis": analysis,
            "worker_results": worker_results,
            "synthesis_input": synthesis_input,
            "synthesis": synthesis,
        }


if __name__ == "__main__":
    ORCHESTRATOR_PROMPT = """
Analyze this task and break it down into {task_count_hint} distinct approaches:
Task: {task}

Return your response in this format:
<analysis>
Explain your understanding of the task and which variations would be valuable.
Focus on how each approach serves different aspects of the task.
</analysis>

<tasks>
    <task>
    <type>formal</type>
    <description>Write a precise, technical version that emphasizes specifications</description>
    </task>
    <task>
    <type>conversational</type>
    <description>Write an engaging, friendly version that connects with readers</description>
    </task>
</tasks>
"""

    WORKER_PROMPT = """
Generate content based on:
Task: {original_task}
Style: {task_type}
Guidelines: {task_description}

Return your response in this format:
<response>
Your content here, maintaining the specified style and fully addressing requirements.
</response>
"""

    orchestrator = FlexibleOrchestrator(
        orchestrator_prompt=ORCHESTRATOR_PROMPT,
        worker_prompt=WORKER_PROMPT,
    )

    # Experimenting with orchestrator strategy: ask for more subtasks than the
    # prompt's own default example (formal/conversational) suggests, to see
    # how the decomposition changes.
    results = orchestrator.process(
        task="Write a product description for a new eco-friendly water bottle",
        context={
            "target_audience": "environmentally conscious millennials",
            "key_features": ["plastic-free", "insulated", "lifetime warranty"],
        },
        task_count_hint="3-4",
    )
