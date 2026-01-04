# Reverse-Engineering Manus AI (kind of) 
### Why some teams ship Manus-class systems and others don't

The current "AI agent" landscape shows a severe engineering capability discrepancy. Some teams struggle to deliver systems that remain robust outside controlled demonstrations, while others ship agents that reliably plan, use tools, recover from errors, maintain long-horizon state, and generate consistent artifacts.

This post is my attempt to reverse-engineer how elite teams think about agentic systems design and what actually goes into building something at the level of Manus. These insights come from a mix of research, personal experience building agentic systems, learning from incredibly sharp people in my network, and quite frankly, educated guesses.

**Note**: I'm obviously not going to clone Manus, that's not the point. But the principles here should give anyone interested in the hard parts of AI engineering a solid starting point, and at minimum, a deeper appreciation for the sheer amount of deliberate design work behind these systems.



## 1. Why most agent projects fail

Agentic systems amplify both strengths and weaknesses of engineering practice. In single-turn chat, deficiencies in state handling, observability, and tool reliability can be partially masked. In multi-step, tool-driven environments, these deficiencies become dominant failure modes, namely compounding error across steps, uncontrolled retries, prompt growth, and premature termination.

Systems that appear "state of the art" in agent behavior tend to share:

* a supervised execution loop with explicit control logic
* constrained and deterministic tool surfaces
* durable external state (files, event logs, retrieval stores)
* measurable budgets and termination criteria
* traceability (logging) and reproducibility (replay)
* evaluation harnesses for regression control

The remainder of this post operationalizes these properties into an implementable design.



## 2. The right mental model: LLM as policy, loop as runtime

A robust agent can be modeled as an LLM embedded within a supervisory process that iteratively selects and executes actions:

1. Construct a view of current state (goal, constraints, partial artifacts, recent observations).
2. Produce exactly one next action under a formal action contract.
3. Execute that action via tools in a controlled environment.
4. Record the observation as immutable state.
5. Continue until a controller-defined stopping condition or budget exhaustion.

The model is a **policy over actions**. The process loop is the **runtime** that governs safety, determinism, and completion.



## 3. Layered architecture

In resilient agent stacks, responsibilities are separated into independently evolvable layers:

* **Input normalization / routing**: transforms user requests into structured goals, constraints, and task modes.
* **Loop controller**: owns phases, budgets, retries, and termination.
* **Tool execution layer**: sandboxed code execution, browser automation, API wrappers.
* **Context and memory layer**: typed events, summarization checkpoints, retrieval injection, file-based memory.
* **Monitoring and evaluation**: trace capture, replay, metrics, and regression suites.

This modularity reduces the likelihood that prompt text becomes the sole control mechanism.



## 4. State: typed events, not chat transcripts

### 4.1 Event schema

Raw chat transcripts are not a stable state representation for long-horizon systems because they conflate intent, plan, actions, and observations. A typed event stream makes these roles explicit and supports controlled compression.

```python
from dataclasses import dataclass
from typing import Literal, Any
from time import time

EventType = Literal["user", "plan", "action", "observation", "knowledge", "summary"]

@dataclass(frozen=True)
class Event:
    t: EventType
    ts: float
    content: str
    meta: dict[str, Any] | None = None

def record(t: EventType, content: str, **meta) -> Event:
    return Event(t=t, ts=time(), content=content, meta=meta or None)
```

### 4.2 Append-only history

A high-leverage invariant is **append-only history**, meaning events are never edited or reordered. Summaries are appended as additional events rather than rewriting older content. This yields auditability, simplifies replay, and makes failures attributable to specific actions and observations.



## 5. Memory: the filesystem is your friend

High-performing agent systems externalize intermediate state into durable artifacts rather than expanding prompts indefinitely. A practical workspace convention:

```
workspace/
 ├── todo.md              # plan + checkboxes
 ├── notes/               # intermediate reasoning and research
 ├── artifacts/           # draft and final outputs
 └── runs/<run_id>/       # trace, prompts, actions, metrics
```

This enables bounded prompt growth, human inspectability, and deterministic recovery across restarts.



## 6. Actions: code as the universal tool call

### 6.1 Why code beats JSON tool calls

Many robust agent stacks implement an action contract where the model emits a small executable program calling a narrow, well-documented tool library. This yields:

* compositional tool use (multi-operation steps)
* explicit error handling and branching
* a single inspectable action payload suitable for replay and audit

### 6.2 A minimal tool surface

```python
from pathlib import Path
import subprocess
import shlex
import requests

class Toolset:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if not str(p).startswith(str(self.root)):
            raise ValueError("Path escapes workspace")
        return p

    def write_text(self, rel: str, text: str) -> dict:
        p = self._safe(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return {"ok": True, "path": rel, "n": len(text)}

    def read_text(self, rel: str, limit: int = 50_000) -> dict:
        p = self._safe(rel)
        data = p.read_text(encoding="utf-8")
        if len(data) > limit:
            data = data[:limit] + "\n…(truncated)…"
        return {"ok": True, "text": data}

    def http_get(self, url: str, timeout: int = 15) -> dict:
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "agent/0.1"})
            r.raise_for_status()
            return {"ok": True, "text": r.text}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def shell(self, cmd: str, timeout: int = 10) -> dict:
        allow = {"ls", "cat", "echo", "mkdir", "touch", "pwd"}
        parts = shlex.split(cmd)
        if not parts or parts[0] not in allow:
            return {"ok": False, "error": f"disallowed: {parts[0] if parts else ''}"}
        p = subprocess.run(
            parts, text=True, capture_output=True, timeout=timeout, cwd=str(self.root)
        )
        return {"ok": p.returncode == 0, "stdout": p.stdout, "stderr": p.stderr}
```

### 6.3 The action contract

Each iteration yields a single function `main(tools)` that performs one meaningful step and returns structured status.

```python
def main(tools):
    plan = "\n".join([
        "- [ ] Define scope and constraints",
        "- [ ] Implement tool surface",
        "- [ ] Build controller loop",
        "- [ ] Add tracing + replay + eval harness",
        "- [ ] Add retrieval injection",
    ])
    tools.write_text("todo.md", plan)
    return {"ok": True, "stage": "plan_initialized"}
```

This contract is intentionally restrictive and bounds per-step behavior and improves traceability.



## 7. Control flow: phases, budgets, and termination

A minimal controller defines:

* phases (plan → execute → verify → finalize)
* budgets (max steps, timeouts, retry caps)
* tool gating (per-phase allowlists)
* termination criteria (plan completion and verification success)

```python
from enum import Enum, auto

class Phase(Enum):
    PLAN = auto()
    ACT = auto()
    VERIFY = auto()
    DONE = auto()

class Controller:
    def __init__(self, max_steps: int = 40, max_retries: int = 3):
        self.phase = Phase.PLAN
        self.steps = 0
        self.retries = 0
        self.max_steps = max_steps
        self.max_retries = max_retries

    def allowlist(self) -> set[str]:
        if self.phase == Phase.PLAN:
            return {"read_text", "write_text"}
        if self.phase == Phase.ACT:
            return {"read_text", "write_text", "http_get", "shell"}
        if self.phase == Phase.VERIFY:
            return {"read_text", "shell"}
        return set()

    def stop(self) -> bool:
        return self.phase == Phase.DONE or self.steps >= self.max_steps

    def transition(self, last_status: dict) -> None:
        if self.phase == Phase.PLAN:
            self.phase = Phase.ACT
        elif self.phase == Phase.ACT and last_status.get("stage") == "ready_to_verify":
            self.phase = Phase.VERIFY
        elif self.phase == Phase.VERIFY and last_status.get("ok"):
            self.phase = Phase.DONE
```

This is deliberately plain as reliability typically comes from unambiguous control rules, not sophisticated prose in prompts.



## 8. Scaling out: Task decomposition 

When tasks become "wide" with many sources to consult, many subtasks to complete, quality often degrades. The problem is typically context dilution/polution rather than reasoning limitations. A single agent juggling research, implementation, and verification in one context window loses coherence as the prompt grows.

The architectural response is decomposition. Crucially, this does not require new abstractions. The same primitives from Sections 4–7 apply recursively:

* each worker runs its own controller with a narrow phase set and tool allowlist
* workers persist outputs as structured artifacts in the shared filesystem
* the orchestrator treats worker outputs as observations, injecting summaries as `knowledge` events
* all workers share the same trace format, enabling unified replay and debugging

### 8.1 Files as coordination layer

Workers do not communicate via message passing or shared memory. They coordinate through the filesystem introduced in Section 5:

```
workspace/
 ├── todo.md                    # orchestrator's task breakdown
 ├── workers/
 │    ├── research/
 │    │    ├── status.json      # {"phase": "DONE", "ok": true}
 │    │    └── findings.md      # structured output
 │    └── implementation/
 │         ├── status.json
 │         └── code/
 └── runs/
      ├── orchestrator/         # orchestrator's trace
      ├── worker-research/      # worker traces (same JSONL format)
      └── worker-implementation/
```

This yields several properties. Workers can be restarted independently, partial progress survives failures, and debugging reduces to inspecting files rather than reconstructing distributed state.

### 8.2 Worker specs

A worker is fully specified by a narrow interface:

```python
@dataclass
class WorkerSpec:
    name: str                          # e.g., "research", "implementation"
    workspace: Path                    # isolated subdirectory
    allowed_tools: set[str]            # subset of parent's tools
    allowed_phases: set[Phase]         # typically {PLAN, ACT, VERIFY}
    max_steps: int                     # tighter budget than orchestrator
    goal: str                          # injected into worker's initial context
```

The orchestrator spawns workers by instantiating a `Controller` with the spec's constraints:

```python
def spawn_worker(spec: WorkerSpec, toolset: Toolset, model_call: Callable) -> dict:
    """Run a worker to completion and return its status."""
    worker_root = spec.workspace / spec.name
    worker_tools = Toolset(worker_root)
    
    # Worker gets its own trace
    tracer = TraceLogger(worker_root / "trace.jsonl")
    traced_tools = RecordingToolset(worker_tools, tracer)
    
    # Narrower controller
    ctrl = Controller(max_steps=spec.max_steps)
    events: list[Event] = [record("user", spec.goal)]
    
    while not ctrl.stop():
        if ctrl.phase not in spec.allowed_phases:
            ctrl.transition({"ok": True})  # skip disallowed phases
            continue
            
        # Standard loop from Section 6-7
        action_code = model_call(events, ctrl.allowlist() & spec.allowed_tools)
        result = execute_action(action_code, traced_tools)
        events.append(record("observation", json.dumps(result)))
        ctrl.steps += 1
        ctrl.transition(result)
    
    # Persist final status for orchestrator
    status = {"phase": ctrl.phase.name, "ok": ctrl.phase == Phase.DONE}
    worker_tools.write_text("status.json", json.dumps(status))
    return status
```

### 8.3 The orchestrator

The orchestrator is itself an agent. Its action space includes spawning workers:

```python
def main(tools):
    # Check if research worker has completed
    status_path = "workers/research/status.json"
    try:
        status = json.loads(tools.read_text(status_path)["text"])
        if status.get("ok"):
            # Inject research findings as knowledge
            findings = tools.read_text("workers/research/findings.md")["text"]
            return {"ok": True, "stage": "research_complete", "inject": findings}
    except:
        pass  # Worker not yet started or not yet done
    
    # Spawn research worker (orchestrator yields control)
    return {
        "ok": True, 
        "stage": "spawn_worker",
        "worker": "research",
        "goal": "Survey recent developments in X. Write findings to findings.md."
    }
```

The orchestrator's controller interprets `spawn_worker` results and manages worker lifecycle. Worker outputs become observations in the orchestrator's event stream, maintaining a single audit trail.

### 8.4 This is not "multi-agent"

The term "multi-agent" often implies autonomous entities with independent goals and negotiation protocols. What I describe here is simpler. A single goal, decomposed into isolated subproblems with constrained execution contexts. The workers are not peers; they are functions called by the orchestrator.

This framing matters for reliability. Workers cannot interfere with each other (isolated workspaces), cannot exceed their budgets (constrained controllers), and cannot access tools outside their allowlist. The orchestrator maintains global coherence by controlling when and how worker outputs enter the shared context.

Treat decomposition as context management and throughput engineering, not as a paradigm shift.



## 9. Replay: making failures reproducible

Failures must be reproducible to be fixable. A replay runner treats an agent run as an executable trace:

* **Record mode:** log every tool call (name + args) and tool result to an append-only trace file.
* **Replay mode:** do not execute tools; return recorded results in the same order and assert call/arg consistency.

### 9.1 Trace format

Each tool call becomes two JSONL records:

```json
{"type":"call","id":2,"tool":"http_get","args":{"url":"https://example.com","timeout":15}}
{"type":"result","id":2,"ok":true,"output":{"ok":true,"text":"<html>..."}}
```

### 9.2 Record and replay wrappers

```python
import json
from pathlib import Path
from typing import Any, Callable

class TraceLogger:
    def __init__(self, path: Path):
        self.path = path
        self.counter = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

class RecordingToolset:
    def __init__(self, base: Toolset, tracer: TraceLogger):
        self.base = base
        self.tracer = tracer

    def _wrap(self, tool: str, fn: Callable, **kwargs) -> Any:
        i = self.tracer.counter
        self.tracer.counter += 1
        self.tracer.append({"type": "call", "id": i, "tool": tool, "args": kwargs})
        try:
            out = fn(**kwargs)
            self.tracer.append({"type": "result", "id": i, "ok": True, "output": out})
            return out
        except Exception as e:
            self.tracer.append({"type": "result", "id": i, "ok": False, "error": repr(e)})
            raise

    def write_text(self, rel: str, text: str) -> dict:
        return self._wrap("write_text", self.base.write_text, rel=rel, text=text)

    def read_text(self, rel: str, limit: int = 50_000) -> dict:
        return self._wrap("read_text", self.base.read_text, rel=rel, limit=limit)

    def http_get(self, url: str, timeout: int = 15) -> dict:
        return self._wrap("http_get", self.base.http_get, url=url, timeout=timeout)

    def shell(self, cmd: str, timeout: int = 10) -> dict:
        return self._wrap("shell", self.base.shell, cmd=cmd, timeout=timeout)

class ReplayToolset:
    def __init__(self, trace_path: Path, strict: bool = True):
        rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        self.calls = {r["id"]: r for r in rows if r["type"] == "call"}
        self.results = {r["id"]: r for r in rows if r["type"] == "result"}
        self.counter = 0
        self.strict = strict

    def _replay(self, tool: str, **kwargs) -> Any:
        i = self.counter
        self.counter += 1
        call = self.calls[i]
        res = self.results[i]
        if self.strict:
            assert call["tool"] == tool, f"Tool mismatch at {i}: expected {call['tool']} got {tool}"
            assert call["args"] == kwargs, f"Arg mismatch at {i}"
        if not res["ok"]:
            raise RuntimeError(res.get("error", "replayed tool error"))
        return res["output"]

    def write_text(self, rel: str, text: str) -> dict:
        return self._replay("write_text", rel=rel, text=text)

    def read_text(self, rel: str, limit: int = 50_000) -> dict:
        return self._replay("read_text", rel=rel, limit=limit)

    def http_get(self, url: str, timeout: int = 15) -> dict:
        return self._replay("http_get", url=url, timeout=timeout)

    def shell(self, cmd: str, timeout: int = 10) -> dict:
        return self._replay("shell", cmd=cmd, timeout=timeout)
```

Replay mode enables deterministic reproduction of tool/loop failures and supports trace-driven regression testing.



## 10. Putting it together: a Manus-style implementation

The preceding sections established architectural principles. From now on I'll translate these into runnable components. The goal is not identity with any proprietary system, but replication of core behavior which is controlled execution loops, tool use, file memory, retrieval injection, and replayable traces.

### 10.1 Infrastructure

**Foundation model(s)**
Install a CodeAct-style model and pin the revision you deploy (for reproducibility). One commonly used open baseline is the CodeActAgent stack.

```bash
git clone https://github.com/xingyaoww/code-act
cd code-act
pip install -r requirements.txt
```

A frequent operational pattern is to use:

* a smaller code-capable model for step-wise execution, and
* a larger model for planning or critique (optional), invoked as a separate module.

**Execution environment**
Run tool execution in a containerized sandbox with a mounted workspace.

```bash
docker run -d --name agent-sandbox \
  -v "$(pwd)/workspace:/opt/agent/workspace" \
  ubuntu:22.04 sleep infinity
```

Provision the runtime:

```bash
docker exec -it agent-sandbox bash -c "\
  apt-get update && \
  apt-get install -y python3 python3-pip nodejs npm git curl && \
  pip install requests playwright selenium beautifulsoup4 chromadb openai && \
  playwright install chromium"
```

(Additional hardening—capabilities dropping, seccomp, egress controls should be treated as production requirements, but is orthogonal to the architectural logic in this post.)



### 10.2 Agent loop skeleton

#### 10.2.1 Tools

A stable tool surface should be small and explicit. The earlier `Toolset` is a viable baseline; a browser tool can be added with Playwright as needed. The key requirement is not breadth, but determinism, observability, and controlled side effects.

#### 10.2.2 The loop

A minimal loop composes:

* typed events
* a controller for phases and budgets
* an action contract (`main(tools)`)
* trace recording and (optionally) replay injection

In practice, the "model call" is a replaceable component. The essential design is the loop and the tool/runtime constraints.



### 10.3 Retrieval (ChromaDB + OpenAI embeddings)

Retrieval should be treated as a **controlled injection layer**: the system decides when to retrieve, persists retrieval results as artifacts, and injects a bounded summary into context as a `knowledge` event. This reduces prompt bloat and improves trace/replay fidelity.

#### 10.3.1 Embeddings

```python
from __future__ import annotations
import os
from typing import List
from openai import OpenAI

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Paid embeddings backend. Version this model name alongside your index.
    Re-index if you change embedding models.
    """
    client = _get_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]
```

#### 10.3.2 ChromaDB store

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Dict, List
import chromadb

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    content: str

def get_collection(persist_dir: str, name: str = "kb_chunks") -> chromadb.Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "schema_version": "v1"}
    )

def index_chunks(persist_dir: str, chunks: Iterable[Chunk]) -> None:
    batch = list(chunks)
    if not batch:
        return

    texts = [c.content for c in batch]
    embs = embed_texts(texts)

    ids = [f"{c.doc_id}::chunk::{c.chunk_id}" for c in batch]
    metadatas = [{"doc_id": c.doc_id, "chunk_id": c.chunk_id} for c in batch]

    col = get_collection(persist_dir)

    # Deterministic re-indexing: remove existing ids if present
    try:
        col.delete(ids=ids)
    except Exception:
        pass

    col.add(ids=ids, documents=texts, embeddings=embs, metadatas=metadatas)

def retrieve(persist_dir: str, query: str, k: int = 5) -> List[Dict]:
    col = get_collection(persist_dir)
    q_emb = embed_texts([query])[0]

    out = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    results: List[Dict] = []
    for idx in range(len(out["ids"][0])):
        results.append({
            "id": out["ids"][0][idx],
            "doc_id": out["metadatas"][0][idx].get("doc_id"),
            "chunk_id": out["metadatas"][0][idx].get("chunk_id"),
            "distance": out["distances"][0][idx],
            "content": out["documents"][0][idx],
        })
    return results
```

#### 10.3.3 Injecting retrieval into the loop

```python
import json

def inject_retrieval(
    tools: Toolset,
    events: list[Event],
    persist_dir: str,
    query: str,
    k: int = 5
) -> None:
    hits = retrieve(persist_dir=persist_dir, query=query, k=k)

    # Persist full payload for auditability and replay
    tools.write_text("notes/retrieval.json", json.dumps(hits, ensure_ascii=False, indent=2))

    # Inject a bounded summary as a knowledge event
    lines = []
    for h in hits:
        snippet = (h["content"] or "").replace("\n", " ")
        lines.append(f"- {h['doc_id']}#{h['chunk_id']} (d={h['distance']:.4f}): {snippet[:220]}")

    events.append(record("knowledge", f"Retrieved context for query: {query}\n" + "\n".join(lines)))
```

**Replay note:** in record mode, retrieval results should be captured in traces (or persisted as artifacts as above). In replay mode, retrieval should return recorded results rather than querying ChromaDB, to preserve determinism when the corpus evolves.



### 10.4 Indexing script

The following script implements deterministic ingestion suitable for production hardening:

* stable `doc_id` derived from path + content hash
* deterministic chunk boundaries (character-based with overlap)
* stable `chunk_id` ordering
* explicit persistence directory for ChromaDB
* embedding model pinned in code/config

#### 10.4.1 build_index.py

```python
#!/usr/bin/env python3
"""Deterministic document indexing for ChromaDB retrieval."""

from __future__ import annotations
import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    content: str

def stable_doc_id(path: Path, content: str) -> str:
    """
    Deterministic doc identifier. In production, you may prefer:
    - a stable corpus id + relative path
    - plus a content hash to detect drift
    """
    h = hashlib.sha256()
    h.update(str(path.as_posix()).encode("utf-8"))
    h.update(b"\n")
    h.update(content.encode("utf-8"))
    return h.hexdigest()[:24]

def chunk_text(text: str, *, chunk_size: int = 1800, overlap: int = 200) -> List[str]:
    """
    Deterministic chunker based on character windows.
    This avoids tokenization dependencies and remains stable across environments.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap

    return chunks

def iter_documents(root: Path, exts: set[str]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def build_chunks(paths: Iterable[Path], chunk_size: int, overlap: int) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for path in paths:
        text = load_text(path).strip()
        if not text:
            continue

        doc_id = stable_doc_id(path, text)
        parts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for i, part in enumerate(parts):
            all_chunks.append(Chunk(doc_id=doc_id, chunk_id=i, content=part))
    return all_chunks

def main() -> None:
    ap = argparse.ArgumentParser(description="Index documents into ChromaDB")
    ap.add_argument("--corpus_dir", required=True, type=str, help="Directory containing documents")
    ap.add_argument("--persist_dir", required=True, type=str, help="ChromaDB persistence directory")
    ap.add_argument("--chunk_size", type=int, default=1800, help="Characters per chunk")
    ap.add_argument("--overlap", type=int, default=200, help="Overlap between chunks")
    ap.add_argument("--ext", action="append", default=[], help="File extensions to index")
    args = ap.parse_args()

    # Default extensions if none provided
    extensions = set(e.lower() for e in args.ext) if args.ext else {".md", ".txt"}

    corpus = Path(args.corpus_dir)
    if not corpus.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus}")

    paths = list(iter_documents(corpus, extensions))
    chunks = build_chunks(paths, chunk_size=args.chunk_size, overlap=args.overlap)

    # Import indexer (assumes rag_chroma.py in same directory or PYTHONPATH)
    from rag_chroma import index_chunks

    index_chunks(args.persist_dir, chunks)
    print(f"Indexed {len(chunks)} chunks from {len(paths)} documents into {args.persist_dir}")

if __name__ == "__main__":
    main()
```

#### 10.4.2 Running it

```bash
python build_index.py \
  --corpus_dir ./knowledge_base \
  --persist_dir ./chroma_store \
  --chunk_size 1800 \
  --overlap 200 \
  --ext .md --ext .txt
```

This produces a persistent ChromaDB index suitable for retrieval injection during agent runs.



## 11. Evaluation and workflow

The architectural choices above imply an engineering workflow that differs from prompt-centric iteration:

1. **Versioned interfaces**: tool schemas, action contracts, event formats, retrieval schema.
2. **Trace-first debugging**: every run produces a trace bundle (prompt view, action payloads, tool calls, artifacts).
3. **Replay-driven regression**: failing traces become tests; CI replays them against new runtime versions.
4. **Metrics beyond "works once"**: completion rate by task family, loop length distribution, retry clusters, and time-to-completion under budget constraints.

This methodology transforms agent engineering from demonstration-driven development into testable systems engineering.



## Conclusion

The gap between fragile agent prototypes and Manus-class systems is largely explained by differences in runtime design and operational discipline. Typed state, externalized memory, constrained action formats, explicit control planes, deterministic replay, and retrieval as a controlled injection layer are feasible with open tooling.

The primary challenge is not discovering new prompting strategies; it is implementing the constraints and development practices that make long-horizon behavior reliable, reproducible, and maintainable under real workloads.:
