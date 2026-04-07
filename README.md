---
title: NetworkDiagnosticsEnv
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /
tags:
  - openenv
  - reinforcement-learning
  - network-diagnostics
  - agent-evaluation
---

# NetworkDiagnosticsEnv

> A simulation-based reinforcement learning environment for training and evaluating AI agents on **autonomous multi-OS network troubleshooting**.

Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) spec. Agents must diagnose and resolve realistic network failures by issuing CLI-style tool calls against a graph-based simulated network topology.

---

## Environment Description

Real-world network outages cost billions. SREs spend hours correlating logs across Linux, Windows, macOS, and Android systems. This environment lets AI agents practice that exact skill — in a fully sandboxed, reproducible simulation.

Each episode presents a broken network. The agent must:
1. Discover available diagnostic tools
2. Issue tool calls (ping, nslookup, traceroute, check_logs, etc.)
3. Identify the root cause
4. Submit a `ResolveAction` with their diagnosis

Reward is given for correctness, efficiency (fewer steps), and tool economy.

---

## Tasks

| Task ID | Name | Difficulty | Description |
|---|---|---|---|
| `dns_failure` | DNS Server Failure | 🟢 Easy | named.conf misconfiguration crashes the DNS server |
| `firewall_block` | Firewall Blocking Traffic | 🟡 Medium | iptables rule silently drops outbound packets |
| `cascading_failure` | Cascading Multi-Hop Failure | 🔴 Hard | BGP peer reset → route loss → DB unreachable → web-svc 502 |

---

## Action Space

| Action Type | Fields | Description |
|---|---|---|
| `ListToolsAction` | — | Discover available tools |
| `CallToolAction` | `tool_name`, `tool_params` | Run a diagnostic tool |
| `ResolveAction` | `root_cause`, `fix_applied` | Submit diagnosis and end episode |
| `NetAction` | `command` | Generic shell command (raw) |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `stdout` | str | Terminal output of the last action |
| `stderr` | str | Error output, if any |
| `available_tools` | list | Tools list (after ListToolsAction) |
| `tool_result` | dict | Structured output from tool call |
| `reward` | float | Per-step reward signal |
| `done` | bool | True when episode ends |

---

## Reward Function

| Component | Weight | Signal |
|---|---|---|
| Root cause correctness | 60% | 1.0 exact match, 0.5 partial keyword match, 0.0 wrong |
| Efficiency | 25% | Penalises steps beyond the ideal minimum |
| Tool economy | 15% | Penalises excessive tool calls |
| Difficulty multiplier | ×1.0–1.5 | Scales score up for harder tasks |

Scores range **0.0 → 1.0**. Pass threshold: **≥ 0.5**.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Execute an action |
| `/state` | GET | Get current episode state |
| `/tasks` | GET | List all tasks + action schema |
| `/grader` | POST | Grade a completed episode |
| `/baseline` | GET | Run deterministic baseline, returns reproducible scores |
| `/schema` | GET | Action/observation schema |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

---

## Setup & Usage

### Run locally

```bash
# Install dependencies
pip install fastapi uvicorn networkx httpx openai

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline
export OPENAI_API_KEY=sk-...
export ENV_URL=http://localhost:7860
python baseline.py
```

### Docker

```bash
docker build -t networkdiagnosticsenv .
docker run -p 7860:7860 networkdiagnosticsenv
```

### Example agent interaction

```python
import httpx

base = "http://localhost:7860"

# Start a medium difficulty episode
obs = httpx.post(f"{base}/reset", json={"scenario_id": "firewall_block", "difficulty": "medium"}).json()
print(obs["observation"]["stdout"])

# Discover tools
r = httpx.post(f"{base}/step", json={"action_type": "ListToolsAction"}).json()

# Ping the router
r = httpx.post(f"{base}/step", json={
    "action_type": "CallToolAction",
    "tool_name": "ping",
    "tool_params": {"target": "internet-router"}
}).json()
print(r["observation"]["stdout"])

# Resolve
r = httpx.post(f"{base}/step", json={
    "action_type": "ResolveAction",
    "root_cause": "firewall_rule_drop",
    "fix_applied": "iptables_flush"
}).json()
print(r["observation"]["stdout"])  # "Resolution submitted. Score: X.XX"

# Grade it
grade = httpx.post(f"{base}/grader", json={
    "scenario_id": "firewall_block",
    "root_cause_submitted": "firewall_rule_drop",
    "steps_taken": 3,
    "tool_cost_sum": -0.3
}).json()
print(grade)
```

---

## Baseline Scores

Run `GET /baseline` for reproducible scores without needing an API key.

| Task | Difficulty | Score |
|---|---|---|
| dns_failure | Easy | ~0.99 |
| firewall_block | Medium | ~1.0 |
| cascading_failure | Hard | ~1.12 (capped at 1.0) |

---

## Project Structure

```
openenvs/
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml        # Dependencies
├── baseline.py           # Inference script (OpenAI API)
├── models.py             # Action, Observation, StepResult models
├── client.py             # NetOSDiagEnv client
├── __init__.py
└── server/
    ├── app.py                  # FastAPI app + all endpoints
    ├── network_environment.py  # Core environment logic
    ├── scenario_generator.py   # 3 task scenarios
    ├── reward_engine.py        # Reward computation
    ├── tool_registry.py        # 6 diagnostic tools
    ├── Dockerfile
    └── requirements.txt
```
