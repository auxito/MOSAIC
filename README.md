# MOSAIC

**M**ulti-agent **O**rchestrated **S**imulation for **A**daptive **I**nterview **C**oaching

---

An event-driven multi-agent system that simulates realistic technical interviews. Not a chatbot wrapper — a full architecture tackling real multi-agent challenges: **memory management, persona consistency, information asymmetry, human-AI switching, and extensible workflows**.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Orchestrator                       │
│              (State Machine + Event Bus)              │
│                                                      │
│  Phase: INIT → RESUME_INPUT → RESUME_MODIFY          │
│         → INTERVIEW_LOOP → EVALUATION → REPORT       │
│                                                      │
│  Event routing: Event → Subscribed Agents → Response  │
└──────────────────────────────────────────────────────┘
        ↕ Event Bus ↕
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌────────────┐
│ Interviewer│ │Interviewee │ │ Evaluator  │ │MemoryManager │ │ResumeMod.  │
│ Adaptive Q │ │Consistency │ │ Privileged │ │Fact extract  │ │ 3 styles   │
│ Follow-up  │ │AI/Human    │ │ Exag. det. │ │Contradiction │ │            │
└────────────┘ └────────────┘ └────────────┘ └──────────────┘ └────────────┘
```

**Core mechanism**: Agents communicate via event bus, not direct calls. MemoryManager emits `CONTRADICTION_FOUND` → Interviewer subscribes and shifts to follow-up questions → Evaluator simultaneously records it as a penalty. This is **emergent behavior**, not hardcoded logic.

## Four-Layer Memory Architecture

| Layer | Purpose | Key Problem Solved |
|-------|---------|-------------------|
| **Privileged** | Per-agent visibility control (MemoryPolicy) | Information asymmetry — evaluator sees both resumes, interviewee only sees the modified one |
| **Semantic** | Structured facts extracted from dialogue | Persona drift — interviewee checks facts before answering to stay consistent |
| **Working** | Dynamic context window composition | Token budget — progressive summarization keeps context within limits |
| **Episodic** | Append-only full dialogue log | Replay & evaluation source data |

## Quick Start

```bash
# Install
cd MOSAIC
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your OpenAI-compatible API key

# Run full interview simulation
python -m mosaic.main --workflow full_interview

# Human practice mode (you answer the questions)
python -m mosaic.main --workflow human_practice --rounds 5

# Resume optimization only
python -m mosaic.main --workflow resume_only
```

## Workflows

| Mode | Phases | Use Case |
|------|--------|----------|
| `full_interview` | Init → Resume → Modify → Interview → Eval → Report | Full AI-vs-AI simulation |
| `human_practice` | Init → Resume → Interview → Eval → Report | You practice answering |
| `resume_only` | Init → Resume → Modify → Report | Just optimize your resume |
| `eval_only` | Init → Resume → Interview → Eval → Report | Interview + evaluation |

## Tech Stack

```
openai          — LLM calls (any OpenAI-compatible API)
pydantic        — Data validation + structured output
jinja2          — Prompt templates
rich            — Terminal UI
tiktoken        — Token counting
python-dotenv   — Environment config
```

## Tests

```bash
python -m pytest tests/ -v
# 40 tests covering memory system, event bus, workflows, and integration
```

## Project Structure

```
MOSAIC/
├── src/mosaic/
│   ├── core/                 # Reusable framework layer
│   │   ├── events.py         # EventType + EventBus
│   │   ├── agent.py          # BaseAgent ABC + MemoryPolicy
│   │   ├── orchestrator.py   # State machine + event routing
│   │   ├── workflow.py       # Phase enum + WorkflowConfig
│   │   └── memory/           # Four-layer memory system
│   ├── agents/               # 6 domain agents
│   ├── participants/         # Human/AI switching (Protocol pattern)
│   ├── prompts/              # Jinja2 templates
│   ├── workflows/            # Phase handler registrations
│   ├── resume/               # Resume data model + parsing
│   ├── llm/                  # OpenAI SDK wrapper
│   └── output/               # Markdown report generation
├── tests/                    # 40 unit + integration tests
├── examples/                 # Sample resume + JD
└── reports/                  # Generated interview reports
```
