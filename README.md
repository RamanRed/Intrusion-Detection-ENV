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

Built for the [OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv). Agents diagnose and resolve realistic network failures by issuing CLI-style tool calls against a graph-based simulated network topology.

---

## Environment Description

Real-world network outages cost billions. SREs spend hours correlating logs across Linux, Windows, macOS, and Android systems. This environment lets AI agents practice that exact skill — in a fully sandboxed, reproducible simulation.

Each episode presents a broken network. The agent must:
1. Discover available diagnostic tools (`ListToolsAction`)
2. Issue tool calls — `ping`, `nslookup`, `traceroute`, `check_logs`, `check_service`, `curl`
3. Identify the root cause of the failure
4. Submit a `ResolveAction` with their diagnosis

Reward is given for **correctness** (did they find the right root cause?), **efficiency** (fewer steps = more reward), and **tool economy** (avoid unnecessary tool calls).

---

## Tasks

| Task ID | Name | Difficulty | Expected Root Cause |
|---|---|---|---|
| `dns_failure` | DNS Server Failure | 🟢 Easy | `dns_misconfiguration` |
| `firewall_block` | Firewall Blocking Traffic | 🟡 Medium | `firewall_rule_drop` |
| `cascading_failure` | Cascading Multi-Hop Failure | 🔴 Hard | `bgp_peer_reset` |

### Task Details

**🟢 dns_failure (Easy)**
The DNS server has crashed due to a misconfiguration in `named.conf`. Hosts can no longer resolve domain names. The agent should detect the DNS resolution failure and trace it back to the crashed `named` service.

**🟡 firewall_block (Medium)**
An `iptables` rule on the internet-router silently drops all outbound packets from `host-a`. Internal traffic still works; only internet access is broken. The agent must distinguish between connectivity symptoms and identify the firewall as the root cause.

**🔴 cascading_failure (Hard)**
A web server is returning 502 errors. The full causal chain: BGP peer reset on `core-router` → route to `db-server` lost → app-server DB connection pool exhausted → `web-svc` returns 502. The agent must trace the dependency chain from symptom back to the true root cause (the BGP reset), ignoring the more obvious downstream symptoms.

---

## Action Space

| Action Type | Fields | Description |
|---|---|---|
| `ListToolsAction` | — | Discover all available diagnostic tools |
| `CallToolAction` | `tool_name: str`, `tool_params: dict` | Run a diagnostic tool on the simulated network |
| `ResolveAction` | `root_cause: str`, `fix_applied: str` | Submit diagnosis and end the episode |
| `NetAction` | `command: str` | Generic shell command (raw) |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `stdout` | str | Terminal output of the last action |
| `stderr` | str | Error output, if any |
| `available_tools` | list | Populated after `ListToolsAction` |
| `tool_result` | dict | Structured output from `CallToolAction` |
| `reward` | float | Per-step reward signal (0.0 on most steps, final score on ResolveAction) |
| `done` | bool | True when episode ends |

---

## Available Tools

| Tool | OS Support | Description | Cost |
|---|---|---|---|
| `ping` | All | ICMP connectivity test | -0.10 |
| `nslookup` | All | DNS name resolution | -0.10 |
| `curl` | All | HTTP request to a service | -0.10 |
| `check_service` | Linux | Systemctl/BGP service status | -0.15 |
| `traceroute` | All | Trace network path to destination | -0.15 |
| `check_logs` | All | Read recent log lines from a host | -0.10 |

---

## Reward Function

| Component | Weight | Signal |
|---|---|---|
| Root cause correctness | 60% | 1.0 = exact match, 0.5 = partial keyword match, 0.0 = wrong |
| Efficiency | 25% | Penalises steps beyond the ideal minimum (3/5/8 for easy/medium/hard) |
| Tool economy | 15% | Penalises excessive tool call cost accumulation |
| Difficulty multiplier | ×1.0–1.5 | Scales total score up for harder tasks |

Scores range **0.0 → 1.0** (hard-capped). Pass threshold: **≥ 0.5**.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Environment info and endpoint list |
| `POST /reset` | POST | Start a new episode |
| `POST /step` | POST | Execute an action |
| `GET /state` | GET | Get current episode state |
| `GET /tasks` | GET | List all tasks with action schema |
| `POST /grader` | POST | Grade a completed episode (deterministic, 0.0–1.0) |
| `GET /baseline` | GET | **Run baseline — no API key needed, always reproducible** |
| `GET /schema` | GET | Action and observation space schema |
| `GET /health` | GET | Health check |
| `GET /docs` | GET | Interactive Swagger UI |

---

## Baseline Scores

### No API Key Required

Run `GET /baseline` at any time for reproducible scores — **no OpenAI key needed**:

```bash
curl https://your-space.hf.space/baseline
```

The `/baseline` endpoint uses a deterministic rule-based expert agent built into the server. The `baseline.py` script also works without an API key for the same reason.

### Expected Scores

| Task | Difficulty | Score | Passed |
|---|---|---|---|
| `dns_failure` | Easy | ~0.97 | ✅ |
| `firewall_block` | Medium | ~1.00 | ✅ |
| `cascading_failure` | Hard | ~1.00 | ✅ |
| **Average** | | **~0.99** | |

---

## Setup & Usage

### Run the baseline (no API key needed)

```bash
# Against the HF Space
curl https://your-space.hf.space/baseline

# Using baseline.py (rule-based agent, no API key)
python baseline.py

# Using baseline.py with an LLM agent (optional)
export OPENAI_API_KEY=sk-...
python baseline.py
```

### Run locally

```bash
# Install dependencies
pip install fastapi uvicorn networkx

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline (no API key needed)
python baseline.py
```

### Docker

```bash
docker build -t networkdiagnosticsenv .
docker run -p 7860:7860 networkdiagnosticsenv

# Test it
curl http://localhost:7860/health
curl http://localhost:7860/baseline
```

---

## Example Agent Interaction

```python
import httpx

base = "http://localhost:7860"

# 1. Start a hard episode
obs = httpx.post(f"{base}/reset", json={
    "scenario_id": "cascading_failure",
    "difficulty":  "hard",
    "os_profile":  "linux"
}).json()

# 2. Discover tools
r = httpx.post(f"{base}/step", json={"action_type": "ListToolsAction"}).json()

# 3. Check web-svc (symptom)
r = httpx.post(f"{base}/step", json={
    "action_type": "CallToolAction",
    "tool_name":   "curl",
    "tool_params": {"url": "http://web-svc/health"}
}).json()
print(r["observation"]["stdout"])  # 502 Bad Gateway

# 4. Trace the chain
r = httpx.post(f"{base}/step", json={
    "action_type": "CallToolAction",
    "tool_name":   "check_service",
    "tool_params": {"host": "core-router", "service": "bgp"}
}).json()
print(r["observation"]["stdout"])  # BGP session IDLE

# 5. Submit diagnosis
r = httpx.post(f"{base}/step", json={
    "action_type": "ResolveAction",
    "root_cause":  "bgp_peer_reset",
    "fix_applied": "restart_bgp_session"
}).json()

# 6. Grade it
grade = httpx.post(f"{base}/grader", json={
    "scenario_id":          "cascading_failure",
    "root_cause_submitted": "bgp_peer_reset",
    "steps_taken":          5,
    "tool_cost_sum":        -0.5
}).json()
print(grade)
# {'score': 1.0, 'passed': True, 'breakdown': {...}}
```

---

## Project Structure

```
openenvs/
├── openenv.yaml               # OpenEnv manifest
├── Dockerfile                 # Container build
├── baseline.py                # Baseline script — no API key needed
├── models.py                  # Action, Observation, StepResult dataclasses
├── client.py                  # NetOSDiagEnv WebSocket client
└── server/
    ├── app.py                 # FastAPI app — all endpoints
    ├── network_environment.py # Core environment logic
    ├── scenario_generator.py  # 3 tasks: easy / medium / hard
    ├── reward_engine.py       # Reward computation (RaR)
    ├── tool_registry.py       # 6 diagnostic tools
    ├── requirements.txt
    └── Dockerfile
```
