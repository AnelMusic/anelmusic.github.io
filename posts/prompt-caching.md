# Prompt Caching for Agentic Systems
> This guide uses OpenAI's implementation as the primary reference, but the underlying principles and techniques are applicable to other model providers offering similar caching mechanisms (e.g., Anthropic, Google, and various self hosted inference frameworks). 

Agentic systems repeatedly send the same lengthy prefix (system prompt, tool schemas, policies, few shot examples) and modify only a small suffix (latest observation, tool output, user message). This characteristic makes such systems particularly sensitive to prompt prefill costs and latency, as the same "program header" is reprocessed on every step. Fortunately, when requests share an exact prompt prefix, the inference provider can reuse previously computed attention key/value tensors for that prefix, thereby reducing both latency (time to first token) and input cost on subsequent calls. On OpenAI specifically, prompt caching activates once the prompt reaches 1,024 tokens, and the cached prefix extends in 128 token increments as longer prefixes are reused. Other providers may have different thresholds, but the core mechanism remains the same.



## Overview

Prompt caching has a single hard requirement:

**Cache hits require exact prefix matches.** Static content must appear first and dynamic content must appear last. This principle applies to tool definitions as well: if tool schemas differ between requests, that portion cannot be cached.

In production, one metric summarizes whether your architecture is behaving correctly:

> **How many prompt tokens were served from cache?**

OpenAI exposes this as `usage.prompt_tokens_details.cached_tokens`. Requests below the threshold will still display the field, but it will simply be zero.

Key thresholds:

* Prompt caching applies automatically for prompts **≥ 1024 tokens**.
* The cached prefix length grows in **128 token increments**.

If the cached fraction is high and stable, your agent architecture can scale economically. If it is low or volatile, you are paying repeated prefill cost on every step, often without realizing it.


## Cost Implications for Agentic Architectures

Single turn chat systems can be surprisingly tolerant of inefficiency. Including a large system prompt or verbose policy section may result in only a mild cost increase. Agentic systems, however, are not tolerant of such inefficiency because they operate iteratively. Consider SomeRandomCompanySupport, which handles "I was charged twice" tickets. A typical resolution involves not one call but a loop:

1. Read ticket and conversation context, decide next action
2. Call `BILLING_lookup_invoice`
3. Incorporate tool output, decide refund versus explanation
4. Possibly call `BILLING_initiate_refund`
5. Produce final customer facing response

Even this relatively simple ticket requires 3 to 5 model calls. Real agent stacks include additional passes (planner/executor separation, verification steps, guardrails, retries, escalation heuristics) and can easily reach 20 to 50 calls for a complex task.

The critical observation is that the new information per step is usually small (a tool result, an observation), while the prefix remains large (instructions, policies, tool schemas). Without caching, the prefill cost is paid repeatedly for the same prefix. With caching, the prefill cost is amortized across steps. This is precisely the regime prompt caching is designed for.


## Mechanism of Prompt Caching

Two concepts are often conflated:

* KV caching within a single generation (standard transformer decoding optimization)
* Prompt caching across requests (cross request reuse of the prefill computation)

This guide concerns the latter.

OpenAI's prompt caching stores the **longest previously computed prefix**, starting at 1,024 tokens and extending in 128 token increments. If a subsequent request begins with that same token sequence, OpenAI can reuse the cached computation for that prefix and compute only the new suffix.

### Cache Retention and Eviction

Default caching is in memory and therefore time sensitive. If there is a long pause between steps, cached prefixes can be evicted. For workloads with pauses (support tickets, asynchronous agent workflows), extended retention is often the difference between occasional benefit and reliable benefit. As per OpenAI:
> When using the in-memory policy, cached prefixes generally remain active for 5 to 10 minutes of inactivity, up to a maximum of one hour. Extended prompt cache retention keeps cached prefixes active for longer, up to a maximum of 24 hours but is available only for certain models.

## The Cache Contract: Prefix Stability and Append Only Semantics

A cache friendly prompt behaves less like a narrative and more like a program whose header must remain stable.

If you rewrite earlier sections (even if the meaning is unchanged), you change the token stream and lose the cache match. In practice, this implies an architectural contract:

* **Stable prefix**: system instructions, policies, tool schemas, examples
* **Append only context**: conversation turns, tool outputs, summaries
* **Delta** (always last): newest observation, tool output, or user message; runtime controls, timestamps

The strongest version of the rule is:

> Step N should be Step N−1 plus appended text.


## Determinism in Tool Schema Definitions

Tools can benefit from prompt caching only if their definitions are **identical** between the requests that are expected to share cached prefixes. "Identical" is not semantic equivalence. It is literal token level identity.

"Stable tool schemas" means your tool definitions must produce the same token stream on every request. This requires both stable ordering and stable serialization. Nondeterministic ordering and serialization are frequently overlooked reasons for cache failure.

### Stable Ordering

If the tools appear as `[A, B, C]` in one request and `[B, A, C]` in another, the prefix differs.

Incorrect approach:

```python
tools = fetch_tools_from_db()  # Order may vary
```

Correct approach:

```python
tools = sorted(fetch_tools_from_db(), key=lambda t: t["name"])  # Deterministic order
```

### Stable Serialization

Even with stable ordering, serialization drift breaks prefix matching: key order, whitespace, optional fields, and formatting all affect tokenization.

These are semantically identical but tokenize differently:


> Note: The JSON snippets below are illustrative. In the OpenAI API, tool definitions are structured objects (e.g., `type: "function"` with a `function` block containing `name`, `description`, and JSON-Schema `parameters`). The point here is token-level determinism.

```json
{"name":"CLI_ls","description":"List files","parameters":{"path":{"type":"string"}}}
```

versus

```json
{
  "description": "List files",
  "parameters": { "path": { "type": "string" } },
  "name": "CLI_ls"
}
```

If you embed JSON into text, canonicalize it:

> We use a canonicalized JSON representation with stable key ordering and whitespace-free serialization to ensure tool schemas produce an identical token sequence across requests. This is sufficient for prompt caching, which only requires token-level determinism, not full RFC canonical JSON compliance.

```python
import json

def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
```

Treat tool schemas as an API contract: versioned, deterministic, stable. If you must modify tool definitions, do so as a deliberate version increment and accept that you are warming a new cache.


## Implementation on OpenAI (Responses API)

At this point, the conceptual requirements should be clear: prompt caching only works when requests share an **identical prefix**. The API does not enforce this for you. It simply rewards you when you get it right and silently penalizes you when you don’t.

From an engineering perspective, there are two responsibilities you cannot avoid:

1. **Construct prompts so that reusable content truly remains reusable**
2. **Measure caching behavior continuously so regressions are visible**

The OpenAI Responses API already supports prompt caching automatically. Your job is to align your prompt construction with the caching model and to verify that reality matches your assumptions.

### 6.1 Instrumentation: treat cache behavior as a first-class signal

Before discussing prompt structure, it’s worth emphasizing instrumentation. Without it, caching failures tend to go unnoticed until costs or latency spike.

OpenAI exposes cache usage directly in the response metadata. A minimal extraction function looks like this:

```python
def cache_stats(resp):
    # `usage` can be either a dict or a typed SDK object depending on the client/version.
    usage = getattr(resp, "usage", None) or {}

    # openai-python often returns pydantic models; normalize to a plain dict if needed.
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()

    ptd = (usage.get("prompt_tokens_details") or {})
    cached = ptd.get("cached_tokens", 0)
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)

    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_hit_rate": (cached / prompt) if prompt else 0.0,
        "completion_tokens": completion,
    }
```

In production, this data should be logged alongside latency (especially time-to-first-token) and request metadata such as agent version and tenant. The pattern you want to see is stability: cache hit rates that remain high across steps of the same agent loop.

If cache hit rate suddenly drops after a deploy, something in your prefix changed—often unintentionally.


### Prompt construction: why “small” changes have large effects

Most cache regressions come from prompt construction that *looks* harmless.

A common example is timestamps.

From a reasoning perspective, timestamps feel useful: “the model should know the current time.” In practice, most agents do not rely on absolute wall-clock time for decision-making. They operate on task state, not time state.

More importantly, timestamps are **volatile**. If placed in the wrong location, they guarantee a cache miss.

#### Timestamp placement

In practice, agents rarely benefit from including timestamps in the prompt at all. If you have a valid reason to include one, treat it as volatile runtime data and append it at the very end of the prompt so it doesn’t break prefix caching.

Incorrect (timestamp modifies the prefix on every request):

```python
prompt = f"now={now_iso}\n" + STATIC_PREFIX + f"\nuser={user_msg}"
```

This forces every request to have a unique first token sequence, even if everything else is identical. From the caching system’s perspective, there is nothing to reuse.

Correct (static content first, dynamic content last):

```python
prompt = STATIC_PREFIX + f"\nuser={user_msg}\nnow={now_iso}"
```

Here, the timestamp exists, but it only affects the suffix. If the rest of the prompt matches a previously cached prefix, the cache remains usable.

This pattern generalizes beyond timestamps: request IDs, experiment flags, debug metadata, and tracing information should *never* appear in the prefix.



### Routing affinity and cache retention

Prompt caching is not just about matching text; it is also about **where** requests land.

OpenAI uses routing heuristics to place requests on machines that are likely to already hold the relevant cached prefix. You can improve the odds of reuse by providing explicit hints.

Two parameters matter here:

* `prompt_cache_key`: encourages routing affinity for related requests
* `prompt_cache_retention`: controls how long cached prefixes are retained

A typical use case is multi-tenant or session-based agents, where repeated requests share a stable prefix but may be spaced apart in time.

```python
from openai import OpenAI
client = OpenAI()

resp = client.responses.create(
    model="gpt-5.1",
    instructions=STATIC_INSTRUCTIONS,
    tools=ALL_TOOLS,
    input=prompt,
    prompt_cache_key=f"tenant:{tenant_id}",
    prompt_cache_retention="24h",
)

print(cache_stats(resp))
```

Some important nuances:

* `prompt_cache_key` does not force caching; it biases routing so that requests with the same key are more likely to hit the same cache.
* Default retention (`in_memory`) is time-limited. If your agent pauses between steps, cached prefixes may be evicted.
* Extended retention (`24h`, where supported) is often necessary for async workflows, customer support tickets, or long-running agent sessions.

A useful mental model is to think of the cache as **warm state**, not persistent storage. If you expect reuse across time gaps, you must opt into longer retention.


### The practical test: does Step N reuse Step N−1?

After implementing the above, you should be able to answer one concrete question:

> Is the prompt for step *N* literally the prompt for step *N−1*, with extra text appended?

If the answer is yes, you’ll usually see a high (and often growing) cached-token count from step to step. It won’t always increase every time: cached tokens typically move in 128-token increments, may stay flat between increments, and can drop to 0 if the cache is evicted or the request is routed differently. If you see frequent drops to 0, look for small formatting, ordering, or tool-schema changes—caching will degrade.

This is why prompt caching often fails not due to misunderstanding the API, but due to subtle violations of immutability in prompt construction.


## Masking Tools on OpenAI: restricting behavior without breaking caching

Tool masking is where many otherwise well-designed agent systems quietly sabotage their cache hit rate.

The motivation is reasonable: at any given step, only a subset of tools is relevant. If the agent is reasoning about billing, you would prefer it not to consider filesystem commands or greeting templates. The naive approach is to remove irrelevant tools from the prompt entirely.

From a caching perspective, this is exactly the wrong thing to do.

### Why dynamic tool removal breaks caching

Tool schemas are typically part of the **stable prefix**. They are large, appear early in the prompt, and change rarely by design. This makes them ideal candidates for caching.

If you dynamically remove tools, you are no longer reusing the same prefix. Even if 90% of the tools are unchanged, the serialized tool list is different, and therefore the prefix token stream is different.

The consequence is severe but silent:

* cache hits drop to zero or near zero,
* latency increases,
* costs rise,
* and nothing in the API response tells you *why*.

This is why masking must be implemented **without mutating the tool list**.



### Soft masking: instructing the model, not enforcing it

The simplest approach is to keep all tools in the prompt and instruct the model which ones are currently allowed.

For example:

```text
AVAILABLE TOOL GROUPS: BILLING_
UNAVAILABLE TOOL GROUPS: CLI_, MATH_, GREETING_
Only call tools from AVAILABLE TOOL GROUPS.
```

This approach has two important properties:

1. The tool list remains stable, so prompt caching works.
2. Enforcement relies on instruction-following rather than guarantees.

Soft masking can be acceptable in exploratory or low-risk contexts, but it has limitations:

* The model may occasionally violate instructions.
* Safety and correctness depend on prompt quality.
* There is no hard guarantee that disallowed tools won’t be selected.

For production agent systems, soft masking is often a stepping stone, not the final solution.



### Hard masking with `allowed_tools`: the cache-safe enforcement mechanism

OpenAI provides a mechanism specifically designed to solve this problem: **allowed tools**.

The key idea is simple:

* You always provide the **full, stable set of tools**.
* At each step, you specify which subset is allowed to be invoked.

This preserves a cacheable prefix while constraining the model’s action space.

#### The anti-pattern 

```python
# BAD: tool list changes per request
available = [t for t in ALL_TOOLS if t["name"].startswith("BILLING_")]
resp = client.responses.create(
    model="gpt-5.1",
    tools=available,
    input="Investigate duplicate charge."
)
```

Here, the tool list itself changes. From the caching system’s perspective, this is a completely different prompt.

#### The correct pattern

```python
def allow_prefix(prefixes):
    return [
        {"type": "function", "name": t["name"]}
        for t in ALL_TOOLS
        if any(t["name"].startswith(p) for p in prefixes)
    ]

resp = client.responses.create(
    model="gpt-5.1",
    tools=ALL_TOOLS,  # stable, cacheable
    tool_choice={
        "type": "allowed_tools",
        "mode": "auto",
        "tools": allow_prefix(["BILLING_"]),
    },
    input="Investigate duplicate charge."
)
```

In this structure:

* The prefix remains identical across steps and sessions.
* The allowed tool subset is a **runtime constraint**, not a structural change.
* Cache hits remain intact.

This is the preferred production pattern.



### `auto` vs `required`: choosing the right enforcement level

The `allowed_tools` mechanism supports different modes, and the choice matters.

* `mode: "auto"`
  The model may call a tool from the allowed set, or respond in natural language.

* `mode: "required"`
  The model must call one of the allowed tools.

The distinction maps naturally to agent architecture:

* Use **auto** during reasoning or planning phases.
* Use **required** during executor phases, where a tool call is mandatory.

For example, after determining that a refund is needed, an executor step might look like:

```python
resp = client.responses.create(
    model="gpt-5.1",
    tools=ALL_TOOLS,
    tool_choice={
        "type": "allowed_tools",
        "mode": "required",
        "tools": allow_prefix(["BILLING_"]),
    },
    input="Issue the appropriate refund."
)
```

This enforces structure without compromising caching.



### Tool groups as an action-space abstraction

Many mature agent systems group tools by function (`CLI_`, `MATH_`, `BILLING_`, etc.). This is not merely organizational—it enables **step-level action-space control**.

Seen through this lens:

* The full tool list defines the agent’s *capabilities*.
* Allowed tool subsets define the agent’s *current action space*.

Masking becomes analogous to action masking in reinforcement learning: the agent still knows what actions exist, but only some are legal at a given time.

This framing aligns well with both:

* OpenAI’s `allowed_tools` mechanism, and
* logit-level masking in self-hosted systems (discussed in the next section).



### The design principle to internalize

If there is one principle to carry forward, it is this:

> **Mask behavior, not structure.**

Structure (tool schemas, order, serialization) must remain stable to enable caching. Behavior (which tools may be used right now) should be controlled through runtime constraints.

Once this distinction is internalized, tool masking stops being a source of cache regressions and becomes a clean, composable part of agent design.


## Logit Level Constraints for Self Hosted Models (Optional)

On hosted APIs token level logit manipulation is typically unavailable. You can guide behavior through prompting and API level controls but you cannot directly change the probability assigned to individual tokens.

In self hosted deployments this limitation disappears. You control the decoding loop which allows direct modification of logits before sampling. This enables the strongest possible form of masking. Disallowed actions are assigned probability minus infinity and therefore cannot be produced by the model. This is a fundamentally different enforcement mechanism.

In API based systems masking constrains what the model is allowed to choose. In self hosted systems logit level masking constrains what the model is able to choose. Instruction based masking assumes cooperation from the model. Even API level constraints such as allowed tools still operate at a semantic level. The model knows that all tools exist but is instructed or guided to only use some of them. Logit level masking removes that ambiguity entirely. From the model’s perspective disallowed actions simply do not exist in the output space. No probability mass is assigned to them and no amount of reasoning can recover them.

This is particularly valuable in self hosted agent systems where tool misuse has real side effects where executor stages must be deterministic or where safety boundaries must be enforced mechanically rather than heuristically.

Many self hosted agent systems adopt a structured output format for tool invocation often JSON. For example

```json
{"tool":"CLI_ls","args":{"path":"/var/log"}}
```

This structure is intentional. It creates a well defined region of the output where action selection occurs. When the model begins emitting the tool field you know exactly which tokens correspond to the action choice. This makes it possible to intervene during decoding and restrict which tool names are valid continuations.

The enforcement does not rely on the model understanding instructions such as do not use a particular tool. It relies on controlling the decoder itself. Below is a simplified illustration using a Hugging Face style logits processor. The goal is to restrict tool selection so that only a predefined set of tool names can be generated once the tool field has begun.

```python
import torch
from transformers import LogitsProcessor

class AllowedToolNameProcessor(LogitsProcessor):
    def __init__(self, tokenizer, allowed_names):
        self.tok = tokenizer
        self.allowed = allowed_names

    def __call__(self, input_ids, scores):
        text = self.tok.decode(input_ids[0], skip_special_tokens=True)
        anchor = '"tool":"'
        if anchor not in text:
            return scores

        prefix = text.split(anchor, 1)[1].split('"', 1)[0]

        allowed_next = set()
        for name in self.allowed:
            if name.startswith(prefix) and len(name) > len(prefix):
                next_piece = name[len(prefix):len(prefix)+1]
                toks = self.tok.encode(next_piece, add_special_tokens=False)
                if len(toks) == 1:
                    allowed_next.add(toks[0])

        mask = torch.full_like(scores, float("-inf"))
        if allowed_next:
            mask[:, list(allowed_next)] = 0.0
            return scores + mask
        return scores
```

Several clarifications are important. This example is conceptual and not production ready. Real tool names tokenize into multiple tokens. Production systems typically combine grammar constraints token prefix tries and logits processors. The core idea remains unchanged. Illegal continuations are removed from the token distribution before sampling. Once removed they cannot be selected.

Earlier sections discussed grouping tools by prefix such as CLI MATH or BILLING and restricting which groups are valid at a given step. Logit level masking is the natural extension of that idea in self hosted systems. At each step the agent’s capabilities remain unchanged since the full tool set exists. The agent’s legal action space is reduced by masking. That reduction is enforced mechanically at decode time. This mirrors action masking in reinforcement learning. The policy network knows about all actions but only a subset is legal in the current state.

It's worth noting that logit level masking does not interfere with prompt caching. Prompt caching operates on the input tokens that form the prompt prefix. Logit level masking operates on the output distribution during decoding. These mechanisms are orthogonal. 

### When logit level masking is justified

Logit level masking introduces additional decoding complexity and engineering overhead. It is typically justified when tool misuse is costly or dangerous when executor stages must be strictly controlled or when determinism and reproducibility matter more than flexibility. For many teams API level constraints such as allowed tools are sufficient. For teams operating self hosted models with tight safety or correctness requirements logit level constraints become a natural next step.



## Diagnosing Cache Failures

If `cached_tokens` is 0 when hits are expected, the usual causes are:

1. The prompt is under 1,024 tokens (ineligible for caching).
2. Prefix drift: timestamps, IDs, flags, ordering changes, or serialization changes.
3. Tools differ between requests (order, fields, or formatting).
4. Cache eviction due to inactivity (in memory retention).
5. Hotspot behavior: a single prefix and key combination receiving very high query volume can reduce effectiveness.

The practical debugging method is to treat the prompt as a binary artifact: store the exact strings for step 1 and step 2 and diff them. If the diff shows edits near the top, the cache metrics will reflect that.


## Production Considerations

### Required Practices

1. **Freeze the prefix.** Do not place volatile fields near the beginning of the prompt.
2. **Make tool schemas deterministic.** Ensure stable ordering, stable serialization, and versioned changes.
3. **Instrument cache metrics.** Log `cached_tokens`, `prompt_tokens`, and time to first token; alert on unexpected drops.
4. **Mask without mutation.** Keep the tools list stable and restrict tool calls via `allowed_tools`.

### Optional Optimizations

* Use `prompt_cache_key` for routing affinity when multiple tenants or workflows share infrastructure.
* Enable extended retention (`24h`) when supported and warranted by workload characteristics (e.g., bursty or paused workflows).

### Summary of Key Thresholds

* Caching activates at 1,024 tokens and grows in 128 token increments.
* Cache hits require exact prefix matches.
* In memory retention is time limited; extended retention stabilizes workflows with variable request timing.


## Appendix A: Worked Example (SomeRandomCompanySupport)

This appendix demonstrates a single agent resolving one ticket in three steps. The precise token counts are illustrative; the qualitative pattern is the focus.

Assumptions:

* Stable prefix (system instructions, policies, tools): approximately 2,200 tokens
* Each step appends approximately 200 to 600 tokens

### Step 1: Initial Request (Cold Cache)

**Stable prefix** (unchanged across steps):

```
[SYSTEM] You are SomeRandomCompanySupport, a customer support agent...
[POLICY] Verify identity, be concise, never leak private data...
[TOOLS] (all tool schemas, canonical order)
```

**Append only context** (minimal on step 1):

```
[CONTEXT]
Ticket metadata: customer_id=..., plan=Pro, region=EU
Conversation history:
  user: "I was charged twice for order #A123"
```

**Delta**:

```
[DELTA]
Allowed tool groups now: BILLING_
Goal: Determine whether duplicate charge occurred and fix it.
```

**Expected caching behavior**: First request is cold, so `cached_tokens ≈ 0`. This request primes the prefix for subsequent steps.

**Illustrative usage**:

```json
{
  "prompt_tokens": 2600,
  "prompt_tokens_details": { "cached_tokens": 0 },
  "completion_tokens": 180
}
```


### Step 2: Tool Result Appended (Cache Activated)

Assume step 1 triggered a tool call: `BILLING_lookup_invoice(order_id="A123")`.

**Stable prefix** (identical to step 1):

```
[SYSTEM] ...
[POLICY] ...
[TOOLS] ...
```

**Append only context** (tool call and output appended):

```
[CONTEXT]
Ticket metadata: ...
Conversation history:
  user: "I was charged twice for order #A123"

Tool calls so far:
  assistant -> BILLING_lookup_invoice(order_id="A123")
Tool output:
  invoice shows two charges: CHG_001 and CHG_002, same amount, 2 minutes apart
```

**Delta**:

```
[DELTA]
Allowed tool groups now: BILLING_
Goal: Determine whether one charge is pending/voided versus settled; prepare refund if needed.
```

**Expected caching behavior**: A large portion of step 1 should now be cached, because step 2 is step 1 plus appended text.

**Illustrative usage**:

```json
{
  "prompt_tokens": 3200,
  "prompt_tokens_details": { "cached_tokens": 2500 },
  "completion_tokens": 220
}
```


### Step 3: Refund Action and Finalization

Assume step 2 decided to refund the duplicate charge using `BILLING_initiate_refund`.

**Stable prefix** (identical):

```
[SYSTEM] ...
[POLICY] ...
[TOOLS] ...
```

**Append only context**:

```
[CONTEXT]
Ticket metadata: ...
Conversation history:
  user: "I was charged twice for order #A123"

Tool calls so far:
  assistant -> BILLING_lookup_invoice(...)
Tool output:
  two charges confirmed

  assistant -> BILLING_initiate_refund(charge_id="CHG_002", reason="Duplicate charge")
Tool output:
  refund initiated: RFND_8891, ETA 3 to 5 business days
```

**Delta**:

```
[DELTA]
Allowed tool groups now: GREETING_, BILLING_
Goal: Explain resolution, include ETA, ask if anything else is needed.
```

**Expected caching behavior**: Step 2 should be a prefix of step 3, so cached tokens typically increase again.

**Illustrative usage**:

```json
{
  "prompt_tokens": 3900,
  "prompt_tokens_details": { "cached_tokens": 3200 },
  "completion_tokens": 160
}
```


## Appendix B: Common Failure Modes

This section documents the failure modes that most frequently produce the outcome "we enabled caching, but observed no benefit."

### B.1 Volatile Fields at the Beginning of the Prompt

**Incorrect**:

```python
# Timestamp before the stable prefix
prompt = f"now={now_iso}\n" + STATIC_PREFIX + f"\nuser={user_msg}"
```

**Why it fails**: The first tokens change on every request, so the prefix never matches exactly.

**Observed behavior**: Every step appears cold.

```json
{ "prompt_tokens": 3200, "prompt_tokens_details": { "cached_tokens": 0 } }
```

**Solution**: Move volatile fields to the end.

```python
# Timestamp at the end
prompt = STATIC_PREFIX + f"\nuser={user_msg}\nnow={now_iso}"
```

### B.2 Nondeterministic Tool Ordering

**Incorrect**:

```python
# Tool order depends on database return order
tools = fetch_tools_from_db()
```

**Why it fails**: Tools appear early in the prompt and are typically large; reordering them changes the prefix significantly.

**Solution**:

```python
tools = sorted(fetch_tools_from_db(), key=lambda t: t["name"])
```

### B.3 Schema Serialization Drift

**Incorrect**:

```python
# Noncaonical serialization
tools_json = json.dumps(tools, indent=2)
```

Or:

```python
# Dynamic metadata inside schemas
tool["last_updated_at"] = datetime.utcnow().isoformat()
```

**Why it fails**: Token streams differ even when schemas are semantically equivalent.

**Solution**:

```python
def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
```

Additionally, remove dynamic metadata from any cacheable prefix. Version schemas intentionally when changes are necessary.

### B.4 Dynamic Tool Removal

**Incorrect**:

```python
# Changing the tools list changes the prefix
available = [t for t in ALL_TOOLS if is_available(t)]
resp = client.responses.create(tools=available, ...)
```

**Solution**: Keep the tools list stable and gate tool calls via `allowed_tools`.

```python
resp = client.responses.create(
    tools=ALL_TOOLS,
    tool_choice={
        "type": "allowed_tools",
        "mode": "auto",
        "tools": allowed_subset,
    },
    ...
)
```

### B.5 Rewriting Context Instead of Appending

**Why it fails**: If you reorder, reformat, or summarize earlier content in place, you change the prefix token stream. Step N is no longer a prefix extension of step N−1.

**Recommended approach**: Append new turns and tool outputs. If summarization is required, append a summary block rather than rewriting earlier text, unless cache invalidation is acceptable.
