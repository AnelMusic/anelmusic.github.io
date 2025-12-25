# Prompt Caching for Agentic Systems
> This guide uses OpenAI's implementation as the primary reference, but the underlying principles and techniques are applicable to other model providers offering similar caching mechanisms (e.g., Anthropic, Google, and various self hosted inference frameworks). 

Agentic systems repeatedly send the same lengthy prefix (system prompt, tool schemas, policies, few shot examples) and modify only a small suffix (latest observation, tool output, user message). This characteristic makes such systems particularly sensitive to prompt prefill costs and latency, as the same "program header" is reprocessed on every step. Fortunately, when requests share an exact prompt prefix, the inference provider can reuse previously computed attention key/value tensors for that prefix, thereby reducing both latency (time to first token) and input cost on subsequent calls. On OpenAI specifically, prompt caching activates once the prompt reaches 1,024 tokens, and the cached prefix extends in 128 token increments as longer prefixes are reused. Other providers may have different thresholds, but the core mechanism remains the same.

## Table of Contents

1. Overview
2. Cost Implications for Agentic Architectures
3. Mechanism of Prompt Caching
4. The Cache Contract: Prefix Stability and Append Only Semantics
5. Determinism in Tool Schema Definitions
6. Implementation on OpenAI (Responses API)
7. Tool Masking Strategies
8. Logit Level Constraints for Self Hosted Models
9. Diagnosing Cache Failures
10. Production Considerations
11. Appendix A: Worked Example (AcmeSupport)
12. Appendix B: Common Failure Modes


## 1. Overview

Prompt caching has a single hard requirement:

**Cache hits require exact prefix matches.** Static content must appear first; dynamic content must appear last. This principle applies to tool definitions as well: if tool schemas differ between requests, that portion cannot be cached.

In production, one metric summarizes whether your architecture is behaving correctly:

> **How many prompt tokens were served from cache?**

OpenAI exposes this as `usage.prompt_tokens_details.cached_tokens`. Requests below the threshold will still display the field, but it will simply be zero.

Key thresholds:

* Prompt caching applies automatically for prompts **≥ 1024 tokens**.
* The cached prefix length grows in **128 token increments**.

If the cached fraction is high and stable, your agent architecture can scale economically. If it is low or volatile, you are paying repeated prefill cost on every step, often without realizing it.


## 2. Cost Implications for Agentic Architectures

Single turn chat systems can be surprisingly tolerant of inefficiency. Including a large system prompt or verbose policy section may result in only a mild cost increase.

Agentic systems are not tolerant of such inefficiency because they operate iteratively.

Consider AcmeSupport, which handles "I was charged twice" tickets. A typical resolution involves not one call but a loop:

1. Read ticket and conversation context, decide next action
2. Call `BILLING_lookup_invoice`
3. Incorporate tool output, decide refund versus explanation
4. Possibly call `BILLING_initiate_refund`
5. Produce final customer facing response

Even this relatively simple ticket requires 3 to 5 model calls. Real agent stacks include additional passes (planner/executor separation, verification steps, guardrails, retries, escalation heuristics) and can easily reach 20 to 50 calls for a complex task.

The critical observation is that the new information per step is usually small (a tool result, an observation), while the prefix remains large (instructions, policies, tool schemas). Without caching, the prefill cost is paid repeatedly for the same prefix. With caching, the prefill cost is amortized across steps. This is precisely the regime prompt caching is designed for.


## 3. Mechanism of Prompt Caching

Two concepts are often conflated:

* KV caching within a single generation (standard transformer decoding optimization)
* Prompt caching across requests (cross request reuse of the prefill computation)

This guide concerns the latter.

OpenAI's prompt caching stores the **longest previously computed prefix**, starting at 1,024 tokens and extending in 128 token increments. If a subsequent request begins with that same token sequence, OpenAI can reuse the cached computation for that prefix and compute only the new suffix.

### Cache Retention and Eviction

Default caching is in memory and therefore time sensitive. If there is a long pause between steps, cached prefixes can be evicted. For workloads with pauses (support tickets, asynchronous agent workflows), extended retention is often the difference between occasional benefit and reliable benefit.


## 4. The Cache Contract: Prefix Stability and Append Only Semantics

A cache friendly prompt behaves less like a narrative and more like a program whose header must remain stable.

If you rewrite earlier sections (even if the meaning is unchanged), you change the token stream and lose the cache match. In practice, this implies an architectural contract:

* **Stable prefix**: system instructions, policies, tool schemas, examples
* **Append only context**: conversation turns, tool outputs, summaries
* **Delta** (always last): newest observation, tool output, or user message; runtime controls; timestamps

The strongest version of the rule is:

> Step N should be Step N−1 plus appended text, not a rewritten version of Step N−1.


## 5. Determinism in Tool Schema Definitions

This is a frequently overlooked source of cache failures.

Tools can benefit from prompt caching only if their definitions are **identical** between the requests that are expected to share cached prefixes. "Identical" is not semantic equivalence. It is literal token level identity.

"Stable tool schemas" means your tool definitions must produce the same token stream on every request. This requires both stable ordering and stable serialization.

### Stable Ordering

If the tools appear as `[A, B, C]` in one request and `[B, A, C]` in another, the prefix differs.

Incorrect approach:

```python
tools = fetch_tools_from_db()  # Order may vary
````

Correct approach:

```python
tools = sorted(fetch_tools_from_db(), key=lambda t: t["name"])  # Deterministic order
```

### Stable Serialization

Even with stable ordering, serialization drift breaks prefix matching: key order, whitespace, optional fields, and formatting all affect tokenization.

These are semantically identical but tokenize differently:

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

```python
import json

def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
```

Treat tool schemas as an API contract: versioned, deterministic, stable. If you must modify tool definitions, do so as a deliberate version increment and accept that you are warming a new cache.


## 6. Implementation on OpenAI (Responses API)

Prompt caching is automatic, but production systems require two additional components:

1. Prompt construction that preserves a reusable prefix
2. Instrumentation so regressions are detectable

### Logging Cache Usage

```python
def cache_stats(resp):
    usage = resp.usage or {}
    ptd = usage.get("prompt_tokens_details") or {}
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

### Timestamp Placement

Incorrect (modifies prefix on every request):

```python
prompt = f"now={now_iso}\n" + STATIC_PREFIX + f"\nuser={user_msg}"
```

Correct (static content first, dynamic content last):

```python
prompt = STATIC_PREFIX + f"\nuser={user_msg}\nnow={now_iso}"
```

### Routing Affinity and Retention Parameters

OpenAI provides:

* `prompt_cache_key` to influence routing affinity
* `prompt_cache_retention` with `in_memory` (default) and optionally `24h` on supported models

```python
from openai import OpenAI
client = OpenAI()

resp = client.responses.create(
    model="gpt-5.1",
    instructions=STATIC_INSTRUCTIONS,
    tools=ALL_TOOLS,
    input=prompt,
    prompt_cache_key=f"tenant:{tenant_id}",
    prompt_cache_retention="24h",  # or omit / use "in_memory"
)

print(cache_stats(resp))
```


## 7. Tool Masking Strategies

Teams often group tools (`CLI_*`, `MATH_*`, `BILLING_*`) and want to restrict which tools are callable at a given step. A common mistake is implementing this by removing tools dynamically, which changes the prefix and invalidates the cache.

### Soft Masking (Instruction Based)

This approach is cache friendly but relies on instruction following rather than enforcement:

```text
AVAILABLE TOOL GROUPS NOW: BILLING_
UNAVAILABLE GROUPS: CLI_, MATH_, GREETING_
Do not call tools from unavailable groups.
```

### Hard Masking (Recommended): `tool_choice` with `allowed_tools`

This approach preserves a stable tools list (cache friendly) while restricting the callable subset for the current step.

Incorrect (invalidates cache):

```python
available = [t for t in ALL_TOOLS if t["name"].startswith("BILLING_")]
resp = client.responses.create(model="gpt-5.1", tools=available, input="...")
```

Correct (cache friendly):

```python
def allow_prefix(prefixes):
    return [
        {"type": "function", "name": t["name"]}
        for t in ALL_TOOLS
        if any(t["name"].startswith(p) for p in prefixes)
    ]

resp = client.responses.create(
    model="gpt-5.1",
    tools=ALL_TOOLS,  # Stable, therefore cacheable
    tool_choice={
        "type": "allowed_tools",
        "mode": "auto",     # or "required"
        "tools": allow_prefix(["BILLING_"]),
    },
    input="User says they were charged twice for order A123. Investigate and fix.",
)
```

If the current step requires a tool call (e.g., an executor stage), use `mode: "required"` to enforce a tool call among the allowed subset.


## 8. Logit Level Constraints for Self Hosted Models

On hosted APIs, token level logit manipulation is typically unavailable. On self hosted models, it is possible. This represents the strongest form of masking: disallowed actions receive probability −∞, making them impossible to emit.

A common pattern is to have the model emit JSON tool calls:

```json
{"tool":"CLI_ls","args":{"path":"/var/log"}}
```

During generation of the `"tool"` field, you constrain the next token distribution so only allowed tool names remain possible (often via grammar constraints plus a logits processor). A simplified illustration:

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

This is conceptual (real tool names tokenize across multiple tokens), but it captures the principle: the action space is enforced at the decoder rather than relying on instruction following.


## 9. Diagnosing Cache Failures

If `cached_tokens` is 0 when hits are expected, the usual causes are:

1. The prompt is under 1,024 tokens (ineligible for caching).
2. Prefix drift: timestamps, IDs, flags, ordering changes, or serialization changes.
3. Tools differ between requests (order, fields, or formatting).
4. Cache eviction due to inactivity (in memory retention).
5. Hotspot behavior: a single prefix and key combination receiving very high query volume can reduce effectiveness.

The practical debugging method is to treat the prompt as a binary artifact: store the exact strings for step 1 and step 2 and diff them. If the diff shows edits near the top, the cache metrics will reflect that.


## 10. Production Considerations

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


## Appendix A: Worked Example (AcmeSupport)

This appendix demonstrates a single agent resolving one ticket in three steps. The precise token counts are illustrative; the qualitative pattern is the focus.

Assumptions:

* Stable prefix (system instructions, policies, tools): approximately 2,200 tokens
* Each step appends approximately 200 to 600 tokens

### Step 1: Initial Request (Cold Cache)

**Stable prefix** (unchanged across steps):

```
[SYSTEM] You are AcmeSupport, a customer support agent...
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
