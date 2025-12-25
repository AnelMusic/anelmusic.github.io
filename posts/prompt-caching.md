layout: default
title: The most important metric for Agentic AI

# KV Cache Optimization for Agentic AI

Agentic systems repeatedly send the same long prefix (system prompt, tool schemas, policies, few-shot examples) and only change a small suffix (latest observation, tool output, user message). That makes them uniquely sensitive to **prompt prefill** costs and latency: you are effectively paying to reprocess the same “program header” on every step.

OpenAI’s **Prompt Caching** is the primary systems mechanism that amortizes this cost. When requests share an **exact prompt prefix**, OpenAI can reuse previously computed attention key/value tensors for that prefix, reducing both latency (time-to-first-token) and input cost on subsequent calls. Prompt caching begins once the prompt reaches **1,024 tokens**, and the cached prefix extends in **128-token increments** as longer prefixes are reused.

This document is written as an engineering chapter: a mental model first, then implementation constraints, then operational patterns. We will keep one concrete running example—AcmeSupport, a tool-using customer support agent—so the ideas remain grounded and testable.


## Table of Contents

1. Executive Summary
2. The Prefill Tax: why agents amplify cost
3. What Prompt Caching Actually Does
4. The Cache Contract: stable prefix, append-only suffix
5. Determinism: “stable tool schemas” explained (order + serialization)
6. OpenAI Implementation (Responses API, GPT-5.x / GPT-4.1)
7. Masking tools on OpenAI (soft → hard → best practice)
8. Logit-level masking (self-hosted / Manus-style)
9. Debugging: why your cache hit rate is secretly zero
10. Production checklist
11. Quick reference
12. Appendix: AcmeSupport walkthrough (3 steps)
13. Appendix: Anti-pattern gallery (how cache hits collapse)


## 1) Executive Summary: the one metric that predicts your bill

Prompt caching has a single hard requirement:

**Cache hits require exact prefix matches.** Static content must come first; dynamic content must come last. This applies to tool definitions as well: if tool schemas differ between requests, that portion cannot be cached.

In production, one metric summarizes whether your architecture is behaving:

> **How many prompt tokens were served from cache?**

OpenAI exposes this as `usage.prompt_tokens_details.cached_tokens`. Requests below the threshold will still show the field—it will simply be zero.

Key thresholds:

* Prompt caching applies automatically for prompts **≥ 1024 tokens**.
* The cached-prefix length grows in **128-token increments**.

If the cached fraction is high and stable, you have an agent architecture that can scale economically. If it is low or volatile, you are paying repeated prefill cost on every step—often without realizing it.


## 2) The Prefill Tax: why agents amplify cost

Single-turn chat systems can be surprisingly tolerant of waste. You might include a large system prompt or a verbose policy section, and it shows up as a mild cost increase.

Agents are not tolerant, because they repeat.

Consider AcmeSupport, which handles “I was charged twice” tickets. A typical resolution is not one call—it is a loop:

1. Read ticket and conversation context, decide next action
2. Call `BILLING_lookup_invoice`
3. Incorporate tool output, decide refund vs explanation
4. Possibly call `BILLING_initiate_refund`
5. Produce final customer-facing response

Even this “simple” ticket becomes 3–5 model calls. Real agent stacks include additional passes (planner/executor separation, verification steps, guardrails, retries, escalation heuristics) and can easily reach 20–50 calls for a complex task.

The critical observation is that the new information per step is usually small (a tool result, an observation), while the prefix remains large (instructions, policies, tool schemas). Without caching, you pay the prefill cost repeatedly for the same prefix. With caching, you amortize the prefill cost across steps. This is precisely the regime prompt caching is built for.


## 3) What Prompt Caching Actually Does

There are two notions people often mix:

* KV caching within a single generation (standard transformer decoding optimization)
* Prompt caching across requests (cross-request reuse of the prefill computation)

This guide is about the second.

OpenAI’s prompt caching caches the **longest previously computed prefix**, starting at 1,024 tokens and extending in 128-token increments. If a later request begins with that same token sequence, OpenAI can reuse the cached computation for that prefix and compute only the new suffix.

### Retention: why cache hits disappear in staging

Default caching is in-memory and therefore time-sensitive. If there is a long pause between steps, cached prefixes can be evicted. For workloads with pauses (support tickets, async agent workflows), extended retention is often the difference between “occasionally helpful” and “reliably helpful.”


## 4) The Cache Contract: stable prefix, append-only suffix

A cache-friendly prompt behaves less like a narrative and more like a program whose header must remain stable.

If you rewrite earlier sections (even if the meaning is unchanged), you change the token stream and lose the cache match. In practice, this implies an architectural contract:

* **Stable prefix**: system instructions, policies, tool schemas, examples
* **Append-only context**: conversation turns, tool outputs, summaries
* **Delta** (always last): newest observation/tool output/user message, runtime controls, timestamps

The strongest version of the rule is:

> Step N should be Step N−1 plus appended text, not a rewritten version of Step N−1.


## 5) Determinism: “stable tool schemas” explained (order + serialization)

This is the quiet source of many cache failures.

Tools can benefit from prompt caching only if their definitions are **identical** between the requests that you expect to share cached prefixes. “Identical” is not semantic. It is literal token-level identity.

“Stable tool schemas (stable order + stable serialization)” means your tool definitions must produce the same token stream every time.

### Stable order

If the tools appear as `[A, B, C]` in one request and `[B, A, C]` in another, the prefix differs.

Bad:

```python
tools = fetch_tools_from_db()  # BAD: order may vary
````

Good:

```python
tools = sorted(fetch_tools_from_db(), key=lambda t: t["name"])  # GOOD: stable
```

### Stable serialization

Even with stable ordering, serialization drift breaks prefix matching: key order, whitespace, optional fields, formatting.

These are semantically identical but tokenize differently:

```json
{"name":"CLI_ls","description":"List files","parameters":{"path":{"type":"string"}}}
```

vs

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

Treat tool schemas like an API contract: versioned, deterministic, stable. If you must change tool definitions, do it as a deliberate version bump and accept that you are warming a new cache.


## 6) OpenAI implementation (Responses API): build for reuse, measure for regressions

Prompt caching is automatic, but production systems require two additional pieces:

1. Prompt construction that preserves a reusable prefix
2. Instrumentation so regressions are detectable

### Always log cache usage

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

### Bad → Good #1: timestamp at the top

Bad (changes prefix every request):

```python
prompt = f"now={now_iso}\n" + STATIC_PREFIX + f"\nuser={user_msg}"
```

Good (static first, delta last):

```python
prompt = STATIC_PREFIX + f"\nuser={user_msg}\nnow={now_iso}"
```

### Routing affinity and retention knobs

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


## 7) Masking tools on OpenAI (soft → hard → best practice)

Teams often group tools (`CLI_*`, `MATH_*`, `BILLING_*`) and want to restrict which tools are callable at a given step. The mistake is to implement this by removing tools dynamically: that changes the prefix and destroys caching.

### Soft masking (prompt-only)

This is cache-friendly, but it relies on instruction-following rather than enforcement:

```text
AVAILABLE TOOL GROUPS NOW: BILLING_
UNAVAILABLE GROUPS: CLI_, MATH_, GREETING_
Do not call tools from unavailable groups.
```

### Hard masking (best practice): `tool_choice` with `allowed_tools`

This preserves a stable tools list (cache-friendly) while restricting the callable subset for the current step.

Bad (breaks caching):

```python
available = [t for t in ALL_TOOLS if t["name"].startswith("BILLING_")]
resp = client.responses.create(model="gpt-5.1", tools=available, input="...")  # BAD
```

Good (cache-friendly):

```python
def allow_prefix(prefixes):
    return [
        {"type": "function", "name": t["name"]}
        for t in ALL_TOOLS
        if any(t["name"].startswith(p) for p in prefixes)
    ]

resp = client.responses.create(
    model="gpt-5.1",
    tools=ALL_TOOLS,  # GOOD: stable => cacheable
    tool_choice={
        "type": "allowed_tools",
        "mode": "auto",     # or "required"
        "tools": allow_prefix(["BILLING_"]),
    },
    input="User says they were charged twice for order A123. Investigate and fix.",
)
```

If your step is an “executor” stage where a tool call is mandatory, use `mode: "required"` to force a tool call among the allowed subset.


## 8) Logit-level masking (self-hosted / Manus-style)

On hosted APIs you typically cannot manipulate token-level logits. On self-hosted models you can. This is the strongest form of masking: disallowed actions receive probability −∞, so the model cannot emit them.

A common pattern is to have the model emit JSON tool calls:

```json
{"tool":"CLI_ls","args":{"path":"/var/log"}}
```

Then, during generation of the `"tool"` field, you constrain the next-token distribution so only allowed tool names remain possible (often via grammar constraints plus a logits processor). A simplified illustration:

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

This is conceptual (real tool names tokenize across multiple tokens), but it captures the principle: you enforce the action space at the decoder, rather than asking the model politely.


## 9) Debugging: why your cache hit rate is secretly zero

If `cached_tokens` is 0 when you expected hits, the usual causes are:

1. The prompt is under 1,024 tokens (ineligible).
2. Prefix drift: timestamps, IDs, flags, ordering changes, serialization changes.
3. Tools differ between requests (order, fields, or formatting).
4. Cache eviction due to inactivity (in-memory retention).
5. Hotspot behavior: a single prefix+key combination receiving very high QPS can reduce effectiveness.

The practical debugging method is to treat the prompt like a binary: store the exact strings for step 1 and step 2 and diff them. If the diff shows edits near the top, the cache metrics will reflect that reality.


## 10) Production checklist

A minimal production checklist:

1. Freeze the prefix. Do not put volatile fields near the top.
2. Make tool schemas deterministic: stable order, stable serialization, versioned changes.
3. Instrument `cached_tokens`, `prompt_tokens`, and TTFT; alert on drops.
4. Mask without mutating: keep tools stable, restrict with `allowed_tools`.

Then consider `prompt_cache_key` (routing affinity) and extended retention (`24h`) when supported and warranted by your workload.


## 11) Quick reference

* Caching starts at 1,024 tokens and grows in 128-token increments.
* Cache hits require exact prefix matches.
* In-memory retention is time-limited; extended retention can stabilize bursty or paused workflows.
* Avoid dynamic tool removal; keep tools stable and use `allowed_tools` gating.


# Appendix — AcmeSupport walkthrough: 3 steps, what the prompt looks like, and what `cached_tokens` should do

This appendix shows the same agent resolving a single ticket in three steps. The precise token counts are not the point; the qualitative pattern is.

Assume:

* Stable prefix (system + policies + tools): ~2,200 tokens
* Each step appends ~200–600 tokens

### Step 1 — Cold start: read ticket, decide next action

Stable prefix (unchanged across steps):

```
[SYSTEM] You are AcmeSupport, a customer support agent...
[POLICY] Verify identity, be concise, never leak private data...
[TOOLS] (all tool schemas, canonical order)
```

Append-only context (minimal on step 1):

```
[CONTEXT]
Ticket metadata: customer_id=..., plan=Pro, region=EU
Conversation history:
- user: "I was charged twice for order #A123"
```

Delta:

```
[DELTA]
Allowed tool groups now: BILLING_
Goal: Determine whether duplicate charge occurred and fix it.
```

Expected caching behavior: first request is cold, so `cached_tokens ≈ 0`. The request primes the prefix for subsequent steps.

Illustrative usage:

```json
{
  "prompt_tokens": 2600,
  "prompt_tokens_details": { "cached_tokens": 0 },
  "completion_tokens": 180
}
```


### Step 2 — Append tool result, continue reasoning (cache should kick in)

Assume step 1 triggered a tool call like `BILLING_lookup_invoice(order_id="A123")`.

Stable prefix (identical):

```
[SYSTEM] ...
[POLICY] ...
[TOOLS] ...
```

Append-only context (tool call + tool output appended):

```
[CONTEXT]
Ticket metadata: ...
Conversation history:
- user: "I was charged twice for order #A123"

Tool calls so far:
- assistant -> BILLING_lookup_invoice(order_id="A123")
Tool output:
- invoice shows two charges: CHG_001 and CHG_002, same amount, 2 minutes apart
```

Delta:

```
[DELTA]
Allowed tool groups now: BILLING_
Goal: Determine whether one charge is pending/voided vs settled; prepare refund if needed.
```

Expected caching behavior: a large portion of step 1 should now be cached, because step 2 is step 1 plus appended text.

Illustrative usage:

```json
{
  "prompt_tokens": 3200,
  "prompt_tokens_details": { "cached_tokens": 2500 },
  "completion_tokens": 220
}
```


### Step 3 — Append refund action + confirmation, finalize

Assume step 2 decided to refund the duplicate charge using `BILLING_initiate_refund`.

Stable prefix (identical):

```
[SYSTEM] ...
[POLICY] ...
[TOOLS] ...
```

Append-only context:

```
[CONTEXT]
Ticket metadata: ...
Conversation history:
- user: "I was charged twice for order #A123"

Tool calls so far:
- assistant -> BILLING_lookup_invoice(...)
Tool output:
- two charges confirmed

- assistant -> BILLING_initiate_refund(charge_id="CHG_002", reason="Duplicate charge")
Tool output:
- refund initiated: RFND_8891, ETA 3–5 business days
```

Delta:

```
[DELTA]
Allowed tool groups now: GREETING_, BILLING_
Goal: Explain resolution, include ETA, ask if anything else is needed.
```

Expected caching behavior: step 2 should be a prefix of step 3, so cached tokens typically increase again.

Illustrative usage:

```json
{
  "prompt_tokens": 3900,
  "prompt_tokens_details": { "cached_tokens": 3200 },
  "completion_tokens": 160
}
```


# Appendix — Anti-pattern gallery (how cache hits collapse)

This section shows the failure modes that most often produce “we enabled caching, but it did nothing.”

## Anti-pattern 1: volatile fields inserted near the top

Bad:

```python
# BAD: timestamp before the stable prefix
prompt = f"now={now_iso}\n" + STATIC_PREFIX + f"\nuser={user_msg}"
```

Why it breaks caching: the first tokens change on every request, so the prefix never matches exactly.

How it shows up: every step looks cold.

```json
{ "prompt_tokens": 3200, "prompt_tokens_details": { "cached_tokens": 0 } }
```

Fix: move volatile fields to the end.

```python
# GOOD: timestamp at the end
prompt = STATIC_PREFIX + f"\nuser={user_msg}\nnow={now_iso}"
```

## Anti-pattern 2: tool list order changes

Bad:

```python
# BAD: tool order depends on return order
tools = fetch_tools_from_db()
```

Why it breaks caching: tools are early and large; reordering them changes the prefix significantly.

Fix:

```python
tools = sorted(fetch_tools_from_db(), key=lambda t: t["name"])
```

## Anti-pattern 3: schema serialization drift

Bad:

```python
# BAD: non-canonical serialization
tools_json = json.dumps(tools, indent=2)
```

Or:

```python
# BAD: dynamic metadata inside schemas
tool["last_updated_at"] = datetime.utcnow().isoformat()
```

Why it breaks caching: token streams differ even if the schemas are semantically “the same.”

Fix:

```python
def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
```

Also remove dynamic metadata from any cacheable prefix; version schemas intentionally instead.

## Anti-pattern 4: dynamic tool removal

Bad:

```python
# BAD: changing the tools list changes the prefix
available = [t for t in ALL_TOOLS if is_available(t)]
resp = client.responses.create(tools=available, ...)
```

Fix: keep the tools list stable and gate tool calls via allowed tools.

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

## Anti-pattern 5: rewriting history instead of appending

Why it breaks caching: if you reorder, reformat, or summarize earlier content in-place, you change the prefix token stream. Step N is no longer a prefix-extension of step N−1.

Safer pattern: append new turns and tool outputs. If you must summarize, append a summary block rather than rewriting earlier text, unless you accept cache disruption.


End of document.

```
::contentReference[oaicite:0]{index=0}
```
