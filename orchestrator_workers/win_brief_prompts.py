"""Prompt templates for the Deal Win Brief use case.

"Win Brief" is what an exec asks for
before a call — the short answer to "why do we win this, and what's the
risk." Same orchestrator-workers pattern underneath (see main.py):

  1. Orchestrator reads the deal context and decides which 2-4 angles
     actually matter for THIS deal (pricing objection vs. security review
     vs. stalled buyer produce different angles) rather than always
     generating the same fixed template.
  2. One worker per selected angle produces a tight briefing.
  3. SYNTHESIS_PROMPT (optional next step, not run by the base
     FlexibleOrchestrator) merges the briefings into one document with a
     suggested opening line for the call.

Drop straight into FlexibleOrchestrator:

    from win_brief_prompts import ORCHESTRATOR_PROMPT, WORKER_PROMPT

    orchestrator = FlexibleOrchestrator(
        orchestrator_prompt=ORCHESTRATOR_PROMPT,
        worker_prompt=WORKER_PROMPT,
    )
    results = orchestrator.process(
        task=(
            "Deal: Acme Corp, $120k ARR, enterprise. Competitor Zylo is in "
            "the deal. Buyer is the VP of Engineering. Stated objection: "
            "'Zylo is 30% cheaper.' Deal stage: final approval, security "
            "review just started."
        ),
        context={
            "industry": "fintech",
            "deal_stage": "final approval",
        },
    )
"""

ORCHESTRATOR_PROMPT = """
You are preparing a Deal Win Brief for a sales team ahead of a customer call.

A Win Brief is NOT a generic competitor comparison. It answers one question:
"given exactly what's happening in THIS deal, why do we win, and what's the
risk?" The angles that matter change completely depending on the situation —
a pricing objection needs a different brief than a stalled security review.

Deal context: {task}

Analyze the deal and decide which 2-4 angles are actually worth briefing for
THIS specific deal. Do not default to a fixed checklist. Choose only from
angles that are genuinely relevant given the objection, buyer persona, and
deal stage described above. Typical angles include (pick only what fits,
and feel free to name a more precise angle if the situation calls for it):

- pricing_roi: total cost of ownership, ROI framing, discount strategy
- security_compliance: certifications, data handling, compliance posture
- technical_integration: implementation effort, technical differentiation
- migration_risk: switching cost, timeline, disruption to the buyer
- executive_business_case: strategic fit, risk of inaction, exec sponsor angle

Return your response in this format:

<analysis>
Explain what's actually happening in this deal and why the angles you
picked are the ones that will move it forward. Be specific about what you
are deliberately leaving out and why.
</analysis>

<tasks>
    <task>
    <type>pricing_roi</type>
    <description>Specific instruction for this deal's pricing angle, referencing the actual objection or context above</description>
    </task>
    <task>
    <type>security_compliance</type>
    <description>Specific instruction for this deal's security angle, referencing the actual objection or context above</description>
    </task>
</tasks>
"""

WORKER_PROMPT = """
You are writing one section of a Deal Win Brief for a sales rep who is
about to get on a call. Executives and reps alike read this section in
under a minute, so be concrete and skip filler.

Deal context: {original_task}
Section: {task_type}
Instructions: {task_description}

Write a briefing with exactly these four parts:

1. Key differentiator — the single strongest point for this angle, one sentence.
2. Proof point — a concrete number, case study, or fact backing it up.
3. Rebuttal — a direct response to the specific objection in the deal context.
4. Sharp question — one question the rep can ask the buyer that reframes the conversation in our favor.

Return your response in this format:
<response>
Your four-part briefing here, addressing the deal context directly rather
than generic competitor talking points.
</response>
"""

# Optional next step (see cookbook's "Next Steps" enhancement list):
# merges the worker briefings into one document. Not called automatically
# by FlexibleOrchestrator.process() — wire it in as a final llm_call if
# you want the fully assembled Win Brief instead of separate sections.
SYNTHESIS_PROMPT = """
You are assembling the final Deal Win Brief from the section briefings
below. The reader is a sales rep who has 5 minutes before a customer call,
and possibly an exec skimming it for deal risk.

Deal context: {original_task}

Section briefings:
{worker_results}

Produce one cohesive Win Brief:

1. Opening line — a single sentence the rep can use to open the call,
   grounded in the strongest point across all sections.
2. The sections, condensed — keep each section's differentiator, proof
   point, and rebuttal; drop anything redundant across sections.
3. Biggest risk — the one thing most likely to lose this deal, stated
   plainly, even if none of the sections said it directly.

Return your response in this format:
<response>
The assembled Win Brief here.
</response>
"""
