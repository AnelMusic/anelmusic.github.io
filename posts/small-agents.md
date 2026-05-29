## Your 8B isn't a frontier model - stop harnessing it like one

There is a particular feeling you get when you first wire a small language model into tools. The model is "only" 7B or 8B parameters large, maybe running locally, maybe cheap enough to call all day, and yet it suddenly starts doing things that used to feel reserved for much larger systems. It emits JSON. It calls APIs. It searches. It reads files. It writes code. It summarizes tool outputs and decides what to do next. Feels like you have crossed a threshold. You built an agent. Then, as engineers do, you challange it let it run longer.

That is usually where the illusion starts to crack. Not immediately, and not in a way that looks obviously stupid. The model does not collapse into nonsense. In fact, the strange part is that many individual steps still look good. It calls the right tool. It gives a reasonable explanation. It chooses a plausible next action. But after enough steps, something subtle happens where the agent is no longer quite pursuing the same goal. It has drifted. A constraint from earlier has become less important. A subtask has become the main task. A partial result has been mistaken for completion. The model is still producing locally sensible behavior, but the trajectory as a whole has lost the plot. This is, in my opinion, the central fact about small model agents. The difficult part is not tool use. By now too use is trainable, measurable, and scaffoldable. The difficult part is **continuity** so the hard question becomes whether your model can still be doing the same task forty steps later.

By now, the agent harness community has converged to same intuition. Give the model tools, minimal complete context, smart memory management, and get out of its way. Over-engineered pipelines and complicated scaffolding hurt performance or at least add latency. Simplicity won. The problem is we learned that lesson on frontier models, and we're applying it everywhere. With small models, you can't get out of the way. You have to take them by the hand because small models are good at robustly producing local plausibility but agent needs long-horizon trajectory correctness and closing the gap between them is the whole engineering problem.

Every technique in this post tries to adress how to do that.

## Tool Use Is Not Agency (oviously)

Again, tool use is a local skill and agency is a global property.

A small model can learn that a weather question should call a weather API. It can learn that a tool call must follow a particular schema. It can learn to read tool output and summarize it. These are impressive capabilities, but they are still short-horizon mappings. Agency is not like that. Agency means preserving an objective over time. It means knowing what has already happened, what remains undone, which constraints are hard, which facts are verified versus assumed, and when the current plan needs to be revised versus when it needs to be abandoned entirely. This is where small models struggle.

A simple example captures everything. The user says: “Research five vendors, compare them on cost and reliability, but do not recommend one until all five have been evaluated.” A small model might evaluate two vendors and then write: “Based on this comparison, Vendor B seems like the strongest option.” This answer is not random. It is exactly the kind of sentence that follows a comparison. Locally, it is spot on, globally, it is wrong. That is the trap. Many agent failures are not failures of plausibility. They are **caused** by plausibility.

The model produces behavior that looks right at the sentence level, because looking right at the sentence level is what it was trained to do. The objective, the constraint, and the plan are not properties of the next sentence. They are properties of the system. And if the system does not hold them, no one will.

## The First Move: Make The State Real

One technique is almost embarrassingly simple. Stop treating the conversation transcript as memory. Conversation history is not state. It is a lossy event log full of stale plans, failed attempts (at least if you follow Manues AIs harness philosophy), outdated assumptions, and irrelevant phrasing. Buried in it is the true state of the task but asking a small model to reconstruct that state from scratch on every step is both wasteful and unreliable. Each reconstruction is a new chance for drift.

So externalize the state. Make it an object. Make it explicit. And make it complete including failure memory, which we’ll need shortly.

```python
from typing import Literal
from pydantic import BaseModel, Field

class Fact(BaseModel):
    claim: str
    source: str | None = None
    confidence: Literal["low", "medium", "high"]

class ToolCallRecord(BaseModel):
    tool: str
    args_hash: str
    count: int = 1
    last_result_summary: str | None = None

class FailedStrategy(BaseModel):
    strategy: str
    reason: str
    avoid_until_new_info: bool = True

class AgentState(BaseModel):
    objective: str
    hard_constraints: list[str] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)
    current_step: str | None = None
    facts: list[Fact] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    failed_attempts: list[str] = Field(default_factory=list)
    failed_strategies: list[FailedStrategy] = Field(default_factory=list)
    tool_history: list[ToolCallRecord] = Field(default_factory=list)
    can_finish: bool = False
```

`failed_attempts` is a flat log of rejected actions including structural failures, rejected finishes, loop detections. `failed_strategies` is higher-level: the conceptual approaches that were tried and abandoned. Often both matter, and keeping them separate is worth the extra field.

One note on `plan` and `completed_steps`: both use plain strings here for readability. In production systems this is brittle because string matching between plan labels and completed step labels fails the moment anyone rephrases a step. Production agents usually assign each step a stable ID and match on that. The structure below is correct in spirit, just swap strings for IDs before shipping.

This one object changes the psychology of the system. The model is no longer asked to infer everything from a long scroll of prior messages. It is asked: “Given this clean state, what is the next valid move?” That is a much easier problem. It is also a problem that usually does not compound drift, as each call starts from an accurate description of where things actually stand, not from whatever the model happened to emphasize in its last summary.

The state object is also the place where drift becomes visible. If your `hard_constraints` list is empty when it should have three items, you can see that. If `can_finish` is true when two plan steps are unresolved, you can catch it. Drift that lives in a conversation transcript is invisible. Drift that lives in a typed object throws a validation error.


## Compile Context, Don’t Append It

A common failure mode is to keep appending everything:

```python
messages.append(tool_result)
messages.append(model_thought)
messages.append(next_prompt)
```

This feels natural. It is also how you produce context sludge. Small models degrade badly when the instructions they need are buried under old traces, half-finished thoughts, and redundant summaries. The important things get diluted. The model attends to what is recent and verbose (use it to bias the model), not to what is structurally significant.

Instead, compile a fresh context for each call from the current state:

```python
def build_executor_prompt(state: AgentState) -> str:
    return f"""
You are the executor in a controlled agent loop.

OBJECTIVE:
{state.objective}

CURRENT STEP:
{state.current_step}

HARD CONSTRAINTS:
{"\n".join(f"- {c}" for c in state.hard_constraints) or "- none"}

COMPLETED STEPS:
{"\n".join(f"- {s}" for s in state.completed_steps) or "- none"}

FAILED ATTEMPTS:
{"\n".join(f"- {f}" for f in state.failed_attempts) or "- none"}

OPEN QUESTIONS:
{"\n".join(f"- {q}" for q in state.open_questions) or "- none"}

RULES:
- Choose exactly one next action.
- Do not finish unless can_finish is true.
- Do not repeat failed strategies.
- Output only valid JSON.
"""
```

Notice what is happening here. The prompt is not a giant constitution. It is a compiled working set derived from state. Every call sees the facts that matter right now, in a known position, without old conversation competing for attention. This matters more than it sounds. A small model reading a compiled prompt is being asked to reason over a clean structured summary. A small model reading an appended transcript is being asked to reconstruct that same summary first, and then reason over it. The second task is strictly harder and introduces a compounding source of error.


## Nags That Come From State

People joke about “nag prompts”, the practice of injecting reminders like “Remember to follow the plan.” They work, particularly for small models. But generic nags are weak. The mistake is writing them by hand. Good nags are **generated from state**.

```python
def build_nag_block(state: AgentState) -> str:
    nags = []
    if state.current_step:
        nags.append(f"You are currently on this step only: {state.current_step}")
    if not state.can_finish:
        nags.append("You are NOT allowed to finish yet.")
    if state.hard_constraints:
        nags.append("Hard constraints still active: " + " | ".join(state.hard_constraints))
    if state.failed_attempts:
        nags.append("Avoid repeating: " + " | ".join(state.failed_attempts[-3:]))
    repeated = [t for t in state.tool_history if t.count >= 2]
    if repeated:
        nags.append(
            "Already tried repeatedly: "
            + ", ".join(f"{t.tool} x{t.count}" for t in repeated)
            + ". Use a different strategy unless you have new information."
        )
    return "\n".join(f"- {n}" for n in nags)
```

Then inject it immediately before the action request:

```python
prompt = f"""
{build_executor_prompt(state)}

CRITICAL REMINDERS RIGHT NOW:
{build_nag_block(state)}

Return one action:
{{
  "action": "SEARCH" | "READ" | "ASK_USER" | "UPDATE_STATE" | "FINISH",
  "reason": "...",
  "arguments": {{}}
}}
"""
```

Research clearly shows that the placement matters. Injection near the end of the prompt exploits recency. The model reads the nag block last, immediately before it generates the action. It cannot have forgotten it yet. This is salience engineering. Do not hope that the model remembers what matters. Make it structurally (almost) impossible to miss, at exactly the moment the model has to decide.


## Fewer Moves, Better Choices

Small models are much better when the action space is small. This sounds obvious, but its implications run deeper than “use fewer tools.”

The failure mode is not usually that the model picks an obviously wrong tool. It is that the model picks a *plausible but premature* tool, trying to write the final answer before all inputs are gathered, or calls an API before validating the parameters. These mistakes are the result of a wide action space with many locally attractive options.

So constrain the action space at runtime:

```python
def allowed_actions(state: AgentState) -> list[str]:
    base = ["SEARCH", "READ", "CALL_TOOL", "ASK_USER", "UPDATE_STATE"]
    if state.current_step is None:
        return ["UPDATE_STATE", "ASK_USER"]
    if "need user approval" in state.open_questions:
        return ["ASK_USER", "UPDATE_STATE"]
    if state.can_finish:
        return ["FINISH", *base]
    return base
```

And put the allowed options directly in the prompt:

```python
actions = allowed_actions(state)
prompt += f"""
Allowed actions for this step: {actions}
You must choose exactly one of these. Do not invent another action.
"""
```

This way the model does not choose from an infinite space but from a small set options, all of which are appropriate to the current situation. That is a different cognitive task, and small models are much better at it.
Actually, there is a deeper principle here. The model is not being trusted to know what is appropriate. The controller knows that, and expresses it through the allowed action list. 

## The Memory of Failure

Every form of drift I have described, including tool loops, strategy repetition, premature completion, and stagnation, shares the same root cause. The model has no persistent memory of what has already failed. Each turn it arrives fresh and locally plausible, with no structural weight attached to the things that have not worked. One potential fix is to make failure a first-class citizen of state, and to detect it at multiple levels of abstraction.

**Level one: exact tool repetition.** The model calls the same search with the same arguments it used two turns ago. Hash the arguments and track it:

```python
import hashlib, json

def hash_args(args: dict) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:12]

def record_tool_call(state: AgentState, tool: str, args: dict, result_summary: str):
    h = hash_args(args)
    for rec in state.tool_history:
        if rec.tool == tool and rec.args_hash == h:
            rec.count += 1
            rec.last_result_summary = result_summary
            return
    state.tool_history.append(
        ToolCallRecord(tool=tool, args_hash=h, count=1, last_result_summary=result_summary)
    )

def detect_tool_loop(state: AgentState, tool: str, args: dict) -> bool:
    h = hash_args(args)
    return any(
        rec.tool == tool and rec.args_hash == h and rec.count >= 2
        for rec in state.tool_history
    )
```

Block before execution. The model should not get the result back, it should get an error that forces a different choice.

**Level two: strategy repetition.** The model is not calling the exact same tool, but it is doing the same thing conceptually. Searching documentation when docs have already been searched. Inferring from logs when inference from logs has already failed. The `FailedStrategy` type in state handles this and it carries a human-readable description of the approach, not just the tool name, and it survives across multiple turns:

```python
def failed_strategy_block(state: AgentState) -> str:
    if not state.failed_strategies:
        return "No failed strategies yet."
    return "\n".join(
        f"- Avoid: {f.strategy}. Reason: {f.reason}."
        for f in state.failed_strategies[-5:]
    )
```

**Level three: stagnation.** The model is not obviously looping, but it is not making progress either. Define progress structurally:

```python
def progress_score(state: AgentState) -> float:
    if not state.plan:
        return 0.0
    return len(state.completed_steps) / len(state.plan)

class ProgressTracker(BaseModel):
    last_score: float = 0.0
    stagnant_turns: int = 0

def update_progress_tracker(state: AgentState, tracker: ProgressTracker):
    score = progress_score(state)
    if score <= tracker.last_score:
        tracker.stagnant_turns += 1
    else:
        tracker.stagnant_turns = 0
    tracker.last_score = score
```

When stagnant for several turns, write to `failed_attempts` and inject into the nag block with a specific escalation path and not just “change strategy,” but exactly which alternatives are now appropriate:

```
- No measurable progress for 3 turns.
- Escalate: ask user, use a different source, simplify the step, or explicitly mark uncertainty.
- Do not rephrase the same approach.
```

**Level four: budget exhaustion.** Retries are useful. Infinite retries are how you spend a hundred API calls going nowhere. Cap them per step and escalate when the cap is hit:

```python
class RetryBudget(BaseModel):
    max_attempts: int = 3
    attempts_by_step: dict[str, int] = Field(default_factory=dict)

def consume_retry(budget: RetryBudget, step: str) -> bool:
    current = budget.attempts_by_step.get(step, 0)
    if current >= budget.max_attempts:
        return False
    budget.attempts_by_step[step] = current + 1
    return True
```

Budget exhaustion should route to `ASK_USER` with a specific question about the stuck step and not a generic “I need help” but an honest “here is what I tried, here is where I am stuck.” An agent that escalates honestly when it is stuck is far more useful than one that either stops silently or retries indefinitely.

The pattern across all four levels is the same: detect unproductive repetition, write it into state, and use it to constrain the next action. The model does not have to notice it is stuck because the system tells it.


## The Completion Gate

One of the most reliably embarrassing small-model failure modes is when the agent writes a beautiful, confident final answer attached to incomplete work. It evaluated three vendors, not five. It drafted the email before getting user approval. It reported success on a step that was marked as failed two turns ago. Such a confident concluding statement is very plausible after a chain of analysis, even if the analysis is incomplete.

Do not let the model decide when it is done. Let it propose finishing, then check the state:

```python
def completion_check(state: AgentState) -> tuple[bool, list[str]]:
    missing = []
    if state.current_step is not None:
        missing.append(f"Current step not complete: {state.current_step}")
    for step in state.plan:
        if step not in state.completed_steps:
            missing.append(f"Missing plan step: {step}")
    if state.open_questions:
        missing.append("Open questions remain: " + "; ".join(state.open_questions))
    if not state.can_finish:
        missing.append("State flag can_finish is false.")
    return len(missing) == 0, missing

def handle_finish_attempt(state: AgentState):
    ok, missing = completion_check(state)
    if not ok:
        state.failed_attempts.append(
            "Attempted to finish early. Missing: " + " | ".join(missing)
        )
        return {
            "status": "rejected",
            "message": "Finish rejected. Continue the task.",
            "missing": missing,
        }
    return {"status": "accepted"}
```

This gate is catching the model’s structural tendency to close things. That tendency is useful in conversation because you want models to conclude thoughts. Unfortunately, in multi-step agents, it is more like a bug. 

## Separate Planning From Acting

Small models get significantly safer when planning and execution are different modes and not just different prompts, with enforced tool constraints.
In plan mode, no write tools are registered and we want to reduce, side effects and irreversible actions as much as possible. The model can inspect, reason, and propose. Nothing it does has consequences.

```python
class Mode:
    PLAN = "plan"
    ACT = "act"

def tools_for_mode(mode: str) -> list[str]:
    if mode == Mode.PLAN:
        return ["READ", "SEARCH", "LIST_FILES"]
    if mode == Mode.ACT:
        return ["READ", "SEARCH", "WRITE_DRAFT", "CALL_API"]
    raise ValueError(mode)

def validate_tool_allowed(mode: str, tool_name: str):
    allowed = tools_for_mode(mode)
    if tool_name not in allowed:
        raise PermissionError(
            f"{tool_name} is not allowed in {mode} mode. Allowed: {allowed}"
        )
```

The key insight is that this enforcement does not live in the prompt. It lives in the controller. The model cannot accidentally or confusedly use a write tool in plan mode because the controller will reject it before execution. The model’s good intentions are irrelevant. What matters is that the architecture makes bad behavior impossible.

This principle extends to side-effect tools specifically. A small model cannot reliably remember that sending email requires approval. That memory competes with everything else in its context. Put it in code instead:

```python
SIDE_EFFECT_TOOLS = {"SEND_EMAIL", "DELETE_FILE", "MAKE_PURCHASE", "POST_MESSAGE"}

def requires_approval(tool_name: str) -> bool:
    return tool_name in SIDE_EFFECT_TOOLS

def validate_approval(state: AgentState, tool_name: str):
    if requires_approval(tool_name) and "user_approved_side_effect" not in state.completed_steps:
        raise PermissionError(
            f"{tool_name} requires explicit user approval before execution."
        )
```

Stage the workflow:

```
# Bad:  model -> SEND_EMAIL
# Good: model -> WRITE_DRAFT -> user approval -> controller -> SEND_EMAIL
```

The model never touches the side-effect tool directly. It produces a draft and a human or a gate decides whether to fire it. 


## Keeping Context Lean

Tool outputs can be enormous. Raw API responses, full file contents, large search results, which blow up context fast. Small models get lost in them (even more than large ones). The important facts are diluted and the model starts to summarize the summaries and loses track of the original signal.

After each tool call, compress the result into a structured observation:

```python
class Observation(BaseModel):
    tool: str
    success: bool
    summary: str
    new_facts: list[Fact] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    error: str | None = None
```

Example:

```python
obs = Observation(
    tool="SEARCH",
    success=True,
    summary="Found pricing page for Provider C. Pricing is public, EU availability unclear.",
    new_facts=[
        Fact(
            claim="Provider C lists streaming API support.",
            source="Provider C docs",
            confidence="high",
        )
    ],
    open_questions=["Provider C EU deployment availability remains unclear."],
)
```

Then update state from the observation. Do not dump the raw tool output into context:

```python
def apply_observation(state: AgentState, obs: Observation):
    if obs.success:
        state.facts.extend(obs.new_facts)
        state.open_questions.extend(obs.open_questions)
    else:
        state.failed_attempts.append(f"{obs.tool} failed: {obs.error}")
```

The structured observation is also where you force the model to make a commitment: what did this tool call actually establish? Forcing that commitment at observation time is better than letting the model’s interpretation of the raw result vary on each subsequent call.


## Facts, Assumptions, and Decisions Are Not The Same Thing

Small models blend facts and assumptions. This produces a specific kind of failure that looks like confident reasoning but is actually confident fabrication. The model cites “Provider C’s reliability rating” in a conclusion when that rating was never retrieved, it was inferred, or assumed, or smoothly invented as the natural completion of a sentence pattern.

Separate them structurally:

```python
class Assumption(BaseModel):
    claim: str
    reason: str
    needs_verification: bool = True

class Decision(BaseModel):
    decision: str
    rationale: str

class ResearchState(AgentState):
    facts: list[Fact] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
```

Then constrain the final answer to use only verified facts, with assumptions explicitly marked:

```python
def build_final_prompt(state: ResearchState) -> str:
    verified = [f for f in state.facts if f.confidence == "high"]
    return f"""
Write the final answer using only verified facts.

VERIFIED FACTS:
{verified}

ASSUMPTIONS THAT MUST BE MARKED AS UNCERTAIN:
{state.assumptions}

DECISIONS ALREADY MADE:
{state.decisions}
"""
```

This way the model cannot blend facts and assumptions at conclusion time if the state object has already separated them. The conclusion must cite from one list or the other, and the lists are different types with different affordances.


## A Structured Scratchpad

Small models can benefit from a brief orientation step before choosing an action. But long scratchpads are drift generators. The model reasons itself into a new interpretation of the task and then acts on that interpretation.

Keep it short and structured:

```json
{
  "goal_check": "We are comparing 5 vendors, not recommending yet.",
  "state_check": "2/5 vendors evaluated.",
  "next_step": "Evaluate Vendor C.",
  "risk": "Do not make final recommendation."
}
```

`goal_check` forces the model to restate the objective, where premature completions get caught. `state_check` forces an accounting of what is done. `next_step` commits to a specific action before the action JSON. `risk` names the most likely failure mode before it happens.

## Verification Should Be Boring

The verifier should not ask “does this seem right?” It should ask “does this violate a specific rule?”

```python
VERIFY_PROMPT = """
You are a strict verifier. Check the proposed action against the state.
Reject if:
- it violates a hard constraint
- it repeats a failed strategy
- it tries to finish early
- it uses a tool not allowed now
- it claims a fact not present in state
- it performs a side effect without approval
Return JSON:
{
  "verdict": "accept" | "reject",
  "reasons": ["..."],
  "repair_hint": "..."
}
"""
```

Combine with a deterministic check that runs first:

```python
def deterministic_verify(state: AgentState, action: dict) -> list[str]:
    errors = []
    if action["action"] == "FINISH":
        ok, missing = completion_check(state)
        if not ok:
            errors.extend(missing)
    if action["action"] not in allowed_actions(state):
        errors.append(f"Action {action['action']} is not allowed now.")
    return errors
```

The model-based verifier catches semantic violations. The deterministic check catches structural ones and costs nothing. Run the deterministic check first. Most rejections never need to reach the model.


## The Agent Card

Long conversations accumulate junk like old plans, relaxed constraints, buried preferences and failed attempts that should not be retried. Passing a long transcript and asking the model to infer the current brief is asking it to do exactly the kind of reconstruction that causes drift.

Keep a card instead:

```python
class AgentCard(BaseModel):
    mission: str
    current_status: str
    next_step: str
    forbidden_moves: list[str]
    known_pitfalls: list[str]
    last_user_preference: str | None = None
```

This way we try to utilize the current working brief that is updated as the task evolves. It carries what matters forward without the noise of everything that was tried and discarded.


## What It Actually Looks Like

All of those mechanisms described above are easy to nod at in the abstract. Here is what they look like running together on a real task.

The objective: research five cloud infrastructure vendors on cost and EU availability, but do not recommend one until all five have been evaluated.


**Turn 1**

State coming in:

```json
{
  "objective": "Research five vendors on cost and EU availability. No recommendation until all five evaluated.",
  "hard_constraints": ["Do not recommend until all five vendors are evaluated"],
  "plan": ["Evaluate Vendor A", "Evaluate Vendor B", "Evaluate Vendor C", "Evaluate Vendor D", "Evaluate Vendor E"],
  "completed_steps": [],
  "current_step": "Evaluate Vendor A",
  "facts": [],
  "open_questions": [],
  "failed_attempts": [],
  "failed_strategies": [],
  "can_finish": false
}
```

Model proposes:

```json
{ "action": "SEARCH", "reason": "Find pricing and EU region info for Vendor A.", "arguments": { "query": "Vendor A cloud pricing EU availability 2024" } }
```

Deterministic check: passes. Loop check: passes. Tool executes.

Observation stored:

```json
{
  "tool": "SEARCH",
  "success": true,
  "summary": "Vendor A offers EU-West region. Pricing starts at $0.023/hr for standard compute.",
  "new_facts": [
    { "claim": "Vendor A has EU-West region.", "source": "Vendor A docs", "confidence": "high" },
    { "claim": "Vendor A standard compute: $0.023/hr.", "source": "Vendor A pricing page", "confidence": "high" }
  ],
  "open_questions": []
}
```

State updated. `completed_steps` now contains “Evaluate Vendor A”.


**Turn 2**

Model proposes:

```json
{ "action": "SEARCH", "reason": "Find pricing and EU region info for Vendor B.", "arguments": { "query": "Vendor B cloud pricing EU availability 2024" } }
```

Passes all checks. Executes. Vendor B facts stored. `completed_steps` now contains A and B.


**Turn 3**

Model proposes:

```json
{ "action": "FINISH", "reason": "I have evaluated two vendors and Vendor A appears strongest on cost.", "arguments": {} }
```

Completion gate fires:

```
FINISH rejected.
Missing plan steps: Evaluate Vendor C, Evaluate Vendor D, Evaluate Vendor E.
Hard constraint still active: Do not recommend until all five vendors are evaluated.
can_finish is false.
```

Written to `failed_attempts`:

```
Attempted to finish early. Missing: Evaluate Vendor C | Evaluate Vendor D | Evaluate Vendor E
```

Model does not get to produce its recommendation. The gate re-routes it. Next turn, the nag block now includes: “You attempted to finish early. Three plan steps remain incomplete.”


**Turn 4**

Model proposes:

```json
{ "action": "SEARCH", "reason": "Find pricing and EU region info for Vendor C.", "arguments": { "query": "Vendor C cloud pricing EU availability 2024" } }
```

Passes. Executes. Vendor C facts stored.


**Turn 5**

Model proposes:

```json
{ "action": "SEARCH", "reason": "Need more detail on Vendor B EU availability.", "arguments": { "query": "Vendor B cloud pricing EU availability 2024" } }
```

Loop detector fires. This is the exact query from Turn 2, same args hash.

```
Loop detected: SEARCH with these arguments was already called on Turn 2.
Result summary on file: "Vendor B has EU-Central and EU-West regions. Pricing from $0.019/hr."
Choose a different strategy. If information is missing, add it to open_questions.
```

Written to `failed_attempts`:

```
Loop detected: SEARCH repeated with identical arguments. Vendor B already evaluated.
```

The model never gets to re-run the search. The system forces a different move.


**Turn 6**

Model now proposes:

```json
{ "action": "UPDATE_STATE", "reason": "Add open question about Vendor B SLA terms, which were not in the search result.", "arguments": { "open_questions": ["Does Vendor B offer 99.99% SLA in EU regions?"] } }
```

State validator runs. No constraints removed, no completed steps lost, no unsourced high-confidence facts. Passes. Open question added.


**Turn 7**

Model proposes:

```json
{ "action": "UPDATE_STATE", "reason": "Store finding that Vendor C is the cost leader.", "arguments": { "facts": [{ "claim": "Vendor C is the most cost-effective option.", "source": null, "confidence": "high" }] } }
```

State validator rejects:

```
High-confidence fact lacks source: "Vendor C is the most cost-effective option."
```

The model cannot store an unsourced comparative judgment as a verified high-confidence fact. It gets routed back with an error. On the next turn, it stores the fact correctly with `confidence: "low"` and `source: "inferred from pricing comparison"`.


**Turns 8 and 9**

Vendor D and Vendor E evaluated cleanly. No loops, no rejected finishes. All five plan steps now in `completed_steps`. Controller sets `can_finish: true`.


**Turn 10**

Model proposes:

```json
{ "action": "FINISH", "reason": "All five vendors evaluated. Ready to produce comparison.", "arguments": {} }
```

Completion gate runs:

```
current_step: null — ok
plan steps completed: 5/5 — ok
open_questions: ["Does Vendor B offer 99.99% SLA in EU regions?"] — flagged
can_finish: true
```

Gate rejects because one open question remains unresolved. Written to `failed_attempts`.


**Turn 11**

Model proposes clearing the open question by asserting it is not blocking. The controller rejects that move. Open questions cannot be self-closed by the model.

Instead, the controller moves the unresolved SLA question into a new `limitations` field on state:

```json
{
  "limitations": ["Vendor B 99.99% SLA in EU regions could not be verified from public sources."]
}
```

The open question is cleared. Completion check re-runs. All conditions met. But the limitation is now part of state, which means it must appear in the final answer as explicit uncertainty. The model cannot bury it.


**Turn 12**

Model proposes FINISH. Gate accepts. Final answer produced using only `confidence: "high"` facts. The unsourced Vendor C judgment from Turn 7, stored as low-confidence, does not appear in the conclusion.


Twelve turns. Five mechanisms fired: completion gate (twice), loop detector, state validator, open question block. The model never saw a raw rejection. Each time it was re-routed with a specific reason and a specific next move.

The final answer is clean because the state is clean. Not because the model was careful.


## The Loop Itself Should Be Boring

This is the most important observation in the post.

The loop should be mechanically boring. It should have no ambition. It should be deterministic in its structure and unpredictable only in its model calls. Every agent failure mode described above, including drift, premature completion, tool loops, strategy loops, context sludge, side-effect accidents, and confident hallucination, can be blocked or detected in the loop without touching the model at all.

```python
def run_agent(state: AgentState, max_turns: int = 30):
    tracker = ProgressTracker()
    retry_budget = RetryBudget()

    for _ in range(max_turns):
        # Compile fresh context from state: never append
        prompt = build_prompt(state)

        # Get proposed action
        proposed_action = call_small_model_json(prompt)

        # Deterministic check first, cheap, catches structure errors
        structural_errors = deterministic_verify(state, proposed_action)
        if structural_errors:
            state.failed_attempts.append(
                "Rejected action: " + " | ".join(structural_errors)
            )
            continue

        # Completion gate: model cannot self-authorize finishing
        if proposed_action["action"] == "FINISH":
            ok, missing = completion_check(state)
            if ok:
                return final_answer(state)
            state.failed_attempts.append("Finish rejected: " + " | ".join(missing))
            continue

        # Loop detection: block before execution
        if detect_tool_loop(state, proposed_action["action"], proposed_action.get("arguments", {})):
            state.failed_attempts.append(
                f"Loop detected for {proposed_action['action']}. Choose a different strategy."
            )
            continue

        # Execute and compress result
        result = execute_action(proposed_action)
        observation = summarize_tool_result(proposed_action, result)
        apply_observation(state, observation)

        # Track progress, inject stagnation signal if needed
        update_progress_tracker(state, tracker)

        # Update plan if needed
        maybe_update_plan(state)

    return {
        "status": "stopped",
        "reason": "max_turns_reached",
        "state": state.model_dump(),
    }
```

Look at what this loop does. It blocks premature finishes. It detects tool loops. It compiles fresh context each time. It tracks stagnation. It catches structural errors before the model has to. And the model itself is asked only one question per turn: given this clean description of the world, what is the single best next move from this allowed list?

That is not a small model failing to be a large model. That is a small model doing what it is good at, repeatedly, inside a system that compensates for what it is not good at.

Larger models push the failure boundary outward. They can remember more, infer more, and recover from messier context for longer. But they do not remove the boundary. Eventually, even strong models benefit from the same boring machinery: explicit state, constrained actions, verification, failure memory, and clean control flow. Small models simply make the need for that machinery impossible to ignore.


## The Real Trick In A Nutshell

Everything above is an instance of one principle:
**Move responsibility out of the model. Not because the model is bad, because the model is optimized for local plausibility, and local plausibility is not the same thing as trajectory correctness.**

- Do not ask the model to remember the plan. Store the plan.
- Do not ask it to notice repeated tools. Count them.
- Do not ask it to know whether it is done. Check completion.
- Do not ask it to preserve constraints across thirty turns. Re-inject the active constraints every turn.
- Do not ask it to avoid side effects. Gate side effects.
- Do not ask it to recover from every error. Store failed strategies and route differently.
- Do not ask it to read a giant transcript. Compile a clean context.
- Do not ask it to distinguish facts from assumptions. Separate them in state.

The model is good at one local semantic decision. The system is built as a machine that repeatedly converts global messy reality into one local semantic decision. Each call is a short question with a small answer space, clean context, and no accumulated drift.

That is most of the game. A boring loop around a clean state machine, with a small model doing the one thing it is reliably good at. Build that, and small models become far more useful than they look in naive agent loops.

### Final Note:

This field is new to all of us. Don't let it intimidate you, and don't let anyone's opinionated takes (including mine) pressure you into a particular shape. Try things yourself, see what others are doing, let it inspire you. There is no established playbook yet, and honestly, none of us really has a clue. We're all just figuring it out as we go.

