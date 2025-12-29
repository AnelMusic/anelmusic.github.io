# Prompt Caching for Agentic Systems

> This guide uses OpenAI's implementation as the primary reference, but the underlying principles apply to any provider offering similar caching mechanisms, including Anthropic, Google, and various self-hosted inference frameworks.

Agentic AI systems have a cost structure problem that prompt caching solves elegantly, provided you understand the contract it requires.

Unlike single-turn chat, where an inefficient prompt costs you once, agentic systems pay the same tax on every iteration. A customer support agent resolving a billing dispute might make five model calls. A code assistant debugging an error might make twenty. Each call reprocesses the same instructions, policies, and tool definitions before touching the actual work.

Prompt caching eliminates this redundancy. When consecutive requests share an identical token prefix, the inference provider can reuse previously computed attention states rather than recomputing them from scratch. The result is lower latency, lower cost, and a scaling curve that finally makes economic sense.

But caching has a hard requirement that is easy to violate. Cache hits demand exact prefix matches. One timestamp in the wrong place, one tool schema that serializes differently, and you are back to paying full price on every call. Often you will not even realize it.

This guide covers how to design agent architectures that maintain cache eligibility across steps, how to measure whether you are actually getting cache hits, and how to diagnose the subtle failures that silently destroy your economics.


## The Economics of Iteration

Single-turn chat can absorb a bloated prompt because you pay once and move on. Agentic systems cannot afford that tolerance because they operate in loops.

Consider a customer support agent handling a "I was charged twice" ticket.

1. Read the ticket and conversation context, then decide the next action
2. Call `BILLING_lookup_invoice`
3. Incorporate the tool output, then decide between refund and explanation
4. Call `BILLING_initiate_refund`
5. Compose the final customer-facing response

Five model calls for a routine ticket. Production systems that include planner and executor separation, verification steps, guardrails, retries, and escalation heuristics routinely hit 20 to 50 calls for complex tasks.

The arithmetic is stark. If your prefix is 2,000 tokens and you make 30 calls, you are processing 60,000 input tokens. With caching, you process 2,000 tokens once and pay a fraction of that cost for the remaining 29 calls. Without caching, you pay full price every time.

The new information per step is usually small. A tool result. An observation. A user message. The prefix, however, remains large. Instructions, policies, tool schemas. This is precisely the regime prompt caching is designed for.


## How Prompt Caching Works

Before diving into implementation, it helps to clarify what prompt caching actually is, because it is often confused with something else.

KV caching is a standard transformer optimization that reuses key-value states within a single generation. This happens automatically during decoding and you do not control it.

Prompt caching reuses computation across requests. When two requests share an identical prefix, the provider skips recomputing attention for that prefix and starts from the cached state.

This guide concerns the latter.

OpenAI stores the longest previously computed prefix, starting at 1,024 tokens and extending in 128-token increments. If your next request begins with the same token sequence, OpenAI reuses the cached computation and processes only the new suffix.

### Cache Retention and Eviction

Default caching is in-memory and therefore time-sensitive. If there is a long pause between steps, cached prefixes can be evicted. For workloads with pauses, such as support tickets or asynchronous agent workflows, extended retention is often the difference between occasional benefit and reliable benefit.

OpenAI's documentation states that cached prefixes generally remain active for 5 to 10 minutes of inactivity when using the in-memory policy, with a maximum of one hour. Extended prompt cache retention keeps cached prefixes active longer, up to 24 hours, but is available only for certain models.


## The Architectural Contract

A cache-friendly prompt behaves less like prose and more like a program whose header must remain stable.

If you rewrite earlier sections, even if the meaning is unchanged, you change the token stream and lose the cache match. This implies an architectural contract with three layers.

**Stable prefix.** System instructions, policies, tool schemas, and examples. This content never changes between requests.

**Append-only context.** Conversation turns, tool outputs, and summaries. This content grows monotonically but is never edited.

**Delta.** The newest observation, tool output, or user message. Runtime controls and timestamps. This content is replaced each step and always appears last.

The strongest version of the rule is this. Step N should be Step N minus one with additional text appended. Nothing edited. Nothing reordered.


## Tool Schemas Must Be Token-Identical

Tools can benefit from caching only if their definitions produce the exact same token stream on every request. Identical here means literal token identity, not semantic equivalence.

This has two implications that are easy to miss.

### Stable Ordering

If tools appear as `[A, B, C]` in one request and `[B, A, C]` in another, the prefix differs and caching fails.

```python
# Wrong because order depends on database or dict iteration
tools = fetch_tools_from_db()

# Right because ordering is explicit
tools = sorted(fetch_tools_from_db(), key=lambda t: t["name"])
```

### Stable Serialization

Even with stable ordering, key order, whitespace, and formatting affect tokenization.

These two JSON objects are semantically identical but tokenize differently.

```json
{"name":"CLI_ls","description":"List files"}
```

```json
{
  "description": "List files",
  "name": "CLI_ls"
}
```

If you embed JSON in prompts, canonicalize it.

```python
def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
```

Treat tool schemas as an API contract. Versioned, deterministic, and stable. If you must modify them, do so as a deliberate version increment and accept that you are warming a new cache.


## Implementation Patterns

The conceptual requirements should now be clear. Prompt caching only works when requests share an identical prefix. The API does not enforce this for you. It rewards you when you get it right and silently penalizes you when you do not.

From an engineering perspective, you have two responsibilities.

First, construct prompts so that reusable content truly remains reusable.

Second, measure caching behavior continuously so regressions are visible.

### Instrumentation

Before discussing prompt structure, instrumentation deserves emphasis. Without it, caching failures tend to go unnoticed until costs or latency spike.

OpenAI exposes cache usage directly in the response metadata. A minimal extraction function looks like this.

```python
def cache_stats(resp):
    usage = getattr(resp, "usage", None) or {}

    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()

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

In production, log this data alongside latency, especially time to first token, and request metadata such as agent version and tenant. The pattern you want to see is stability. Cache hit rates that remain high across steps of the same agent loop.

If cache hit rate suddenly drops after a deploy, something in your prefix changed. Often unintentionally.

### Timestamps and Other Volatile Fields

Most cache regressions come from prompt construction that looks harmless.

Timestamps are the most common culprit.

From a reasoning perspective, timestamps feel useful. The model should know the current time. In practice, most agents do not rely on absolute wall-clock time for decision-making. They operate on task state, not time state.

More importantly, timestamps are volatile. Placed in the wrong location, they guarantee a cache miss.

```python
# Wrong because the timestamp contaminates the prefix
prompt = f"now={now_iso}\n" + STATIC_PREFIX + user_msg

# Right because the timestamp only affects the suffix
prompt = STATIC_PREFIX + user_msg + f"\nnow={now_iso}"
```

This pattern generalizes. Request IDs, experiment flags, debug metadata, and tracing information should never appear in the prefix.

### Routing Affinity and Extended Retention

Prompt caching is not just about matching text. It is also about where requests land.

OpenAI uses routing heuristics to place requests on machines that are likely to already hold the relevant cached prefix. You can improve the odds of reuse by providing explicit hints.

Two parameters matter here.

The `prompt_cache_key` parameter encourages routing affinity for related requests.

The `prompt_cache_retention` parameter controls how long cached prefixes are retained.

```python
from openai import OpenAI
client = OpenAI()

resp = client.responses.create(
    model="gpt-5.1",
    instructions=STATIC_INSTRUCTIONS,
    tools=ALL_TOOLS,
    input=prompt,
    prompt_cache_key=f"tenant_{tenant_id}",
    prompt_cache_retention="24h",
)

print(cache_stats(resp))
```

The `prompt_cache_key` does not force caching. It biases routing so that requests with the same key are more likely to hit the same cache.

Default retention is time-limited. If your agent pauses between steps, cached prefixes may be evicted.

Extended retention, where supported, is often necessary for async workflows, customer support tickets, or long-running agent sessions.

Think of the cache as warm state rather than persistent storage. If you expect reuse across time gaps, you must opt into longer retention.


## Tool Masking Without Breaking Caching

Tool masking is where many otherwise well-designed agent systems quietly sabotage their cache hit rate.

The motivation is reasonable. At any given step, only a subset of tools is relevant. If the agent is reasoning about billing, you would prefer it not consider filesystem commands or greeting templates. The naive approach is to remove irrelevant tools from the prompt entirely.

From a caching perspective, this is exactly wrong.

### Why Dynamic Tool Removal Fails

Tool schemas are typically part of the stable prefix. They are large, appear early in the prompt, and change rarely by design. This makes them ideal candidates for caching.

If you dynamically remove tools, you are no longer reusing the same prefix. Even if 90 percent of the tools are unchanged, the serialized tool list is different and therefore the prefix token stream is different.

The consequence is severe but silent. Cache hits drop to zero or near zero. Latency increases. Costs rise. Nothing in the API response tells you why.

### Soft Masking

The simplest approach is to keep all tools in the prompt and instruct the model which ones are currently allowed.

```
AVAILABLE TOOL GROUPS consist of BILLING_
UNAVAILABLE TOOL GROUPS consist of CLI_, MATH_, GREETING_
Only call tools from AVAILABLE TOOL GROUPS.
```

This approach has two properties. The tool list remains stable so prompt caching works. Enforcement relies on instruction-following rather than guarantees.

Soft masking can be acceptable in exploratory or low-risk contexts. For production agent systems, it is often a stepping stone rather than the final solution.

### Hard Masking with allowed_tools

OpenAI provides a mechanism specifically designed to solve this problem.

You always provide the full stable set of tools. At each step, you specify which subset is allowed to be invoked. This preserves a cacheable prefix while constraining the model's action space.

The anti-pattern looks like this.

```python
# Wrong because the tool list changes per request
available = [t for t in ALL_TOOLS if t["name"].startswith("BILLING_")]
resp = client.responses.create(
    model="gpt-5.1",
    tools=available,
    input="Investigate duplicate charge."
)
```

The correct pattern looks like this.

```python
def allow_prefix(prefixes):
    return [
        {"type": "function", "name": t["name"]}
        for t in ALL_TOOLS
        if any(t["name"].startswith(p) for p in prefixes)
    ]

resp = client.responses.create(
    model="gpt-5.1",
    tools=ALL_TOOLS,  # stable and cacheable
    tool_choice={
        "type": "allowed_tools",
        "mode": "auto",
        "tools": allow_prefix(["BILLING_"]),
    },
    input="Investigate duplicate charge."
)
```

The prefix remains identical across steps and sessions. The allowed tool subset is a runtime constraint rather than a structural change. Cache hits remain intact.

### Auto versus Required Mode

The `allowed_tools` mechanism supports different modes.

In auto mode, the model may call a tool from the allowed set or respond in natural language.

In required mode, the model must call one of the allowed tools.

Use auto during reasoning or planning phases. Use required during executor phases where a tool call is mandatory.

### The Principle

If there is one principle to carry forward, it is this. Mask behavior, not structure.

Structure, meaning tool schemas, order, and serialization, must remain stable to enable caching. Behavior, meaning which tools may be used right now, should be controlled through runtime constraints.

Once this distinction is internalized, tool masking stops being a source of cache regressions and becomes a clean, composable part of agent design.


## Logit-Level Constraints for Self-Hosted Models

Skip this section if you use hosted APIs like OpenAI or Anthropic. It applies only to self-hosted deployments where you control the decoding loop.

In self-hosted systems, you can enforce tool restrictions at a level impossible with APIs. You modify the logit distribution directly before sampling. Disallowed tokens receive probability negative infinity. From the model's perspective, those actions do not exist in the output space.

This is fundamentally different from instruction-based masking. API-level constraints such as `allowed_tools` still operate at a semantic level. The model knows all tools exist but is guided to use only some of them. Logit-level masking removes that ambiguity entirely. No probability mass is assigned to disallowed actions and no amount of reasoning can recover them.

Many self-hosted agent systems adopt structured output formats for tool invocation, often JSON.

```json
{"tool":"CLI_ls","args":{"path":"/var/log"}}
```

This structure is intentional. It creates a well-defined region of the output where action selection occurs. When the model begins emitting the tool field, you know exactly which tokens correspond to the action choice. This makes it possible to intervene during decoding and restrict which tool names are valid continuations.

Below is a simplified illustration using a Hugging Face style logits processor. The goal is to restrict tool selection so that only a predefined set of tool names can be generated once the tool field has begun.

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

This example is conceptual rather than production-ready. Real tool names tokenize into multiple tokens. Production systems typically combine grammar constraints, token prefix tries, and logits processors. The core idea remains unchanged. Illegal continuations are removed from the token distribution before sampling. Once removed, they cannot be selected.

Logit-level masking does not interfere with prompt caching. Prompt caching operates on input tokens that form the prompt prefix. Logit-level masking operates on the output distribution during decoding. These mechanisms are orthogonal.

### When to Use Logit-Level Masking

Logit-level masking introduces decoding complexity and engineering overhead. It is typically justified when tool misuse is costly or dangerous, when executor stages must be strictly controlled, or when determinism and reproducibility matter more than flexibility. For many teams, API-level constraints are sufficient. For teams operating self-hosted models with tight safety or correctness requirements, logit-level constraints become a natural next step.


## Diagnosing Cache Failures

If `cached_tokens` is zero when hits are expected, the usual causes fall into five categories.

**Prompt too short.** The prompt is under 1,024 tokens and therefore ineligible for caching.

**Prefix drift.** Timestamps, IDs, flags, ordering changes, or serialization changes have contaminated the prefix.

**Tool differences.** Tools differ between requests in order, fields, or formatting.

**Cache eviction.** Long pauses between steps have caused cached prefixes to be evicted due to in-memory retention limits.

**Hotspot behavior.** A single prefix and key combination receiving very high query volume can reduce effectiveness.

The practical debugging method is to treat the prompt as a binary artifact. Store the exact strings for step one and step two, then diff them. If the diff shows edits near the top, the cache metrics will reflect that.
