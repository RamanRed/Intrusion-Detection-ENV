# NetworkDiagnosticsEnv

An **OpenEnv-compliant** reinforcement-learning environment for evaluating AI agents on autonomous network troubleshooting.

**Phase 1 ✓ · Phase 2 ✓** — 3 grader domains × 3 tasks each = 9 tasks total.

---

## Architecture

```
openenvs/
├── inference.py              # Agent script (LLM + deterministic fallback)
├── Dockerfile                # Container entry point
├── openenv.yaml              # OpenEnv spec (tasks, graders, endpoints)
├── models.py                 # Shared dataclasses (Action, Observation, StepResult…)
└── server/
    ├── app.py                # FastAPI server (all endpoints)
    ├── scenario_generator.py # 9 scenarios, graph topologies, ground truth
    ├── network_environment.py# Episode management, step dispatch, truncation
    ├── reward_engine.py      # 7-dimension multi-criteria reward scoring
    └── tool_registry.py      # 15 diagnostic tools with realistic output
```

---

## Grader Domains

| Grader endpoint | Tasks |
|---|---|
| `POST /grader/connectivity` | `dns_failure` · `dhcp_starvation` · `firewall_block` |
| `POST /grader/infrastructure` | `ntp_drift` · `cascading_failure` · `routing_loop` |
| `POST /grader/distributed` | `split_brain` · `replica_lag` · `job_queue_stall` |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| POST | `/reset` | Start or restart an episode |
| POST | `/step` | Execute one action |
| GET | `/state` | Current episode state |
| GET | `/tasks` | All 9 tasks with grader descriptors |
| GET | `/tasks/{id}` | Single task detail |
| POST | `/grader/connectivity` | Grade a connectivity episode |
| POST | `/grader/infrastructure` | Grade an infrastructure episode |
| POST | `/grader/distributed` | Grade a distributed-systems episode |
| POST | `/grader` | Legacy grader (auto-routes by domain) |
| GET | `/baseline` | Deterministic expert baseline (no API key needed) |
| GET | `/schema` | Action and observation space definitions |
| GET | `/docs` | Swagger UI |

---

## Action Types

```json
{"action_type": "ListToolsAction"}
{"action_type": "CallToolAction", "tool_name": "check_bgp", "tool_params": {"host": "core-router"}}
{"action_type": "ResolveAction",  "root_cause": "bgp_peer_reset", "fix_applied": "restart BGP session"}
{"action_type": "NetAction",      "command": "ip route show"}
```

---

## Available Tools (15)

| Category | Tools |
|---|---|
| Connectivity | `ping`, `traceroute`, `arp_scan` |
| DNS | `nslookup` |
| HTTP | `curl` |
| Service | `check_service` |
| Logs | `check_logs` |
| Firewall | `check_iptables` |
| DHCP | `check_dhcp` |
| Time | `check_ntp` |
| Routing | `check_bgp`, `check_routes` |
| Cluster | `check_cluster` |
| Database | `check_replica` |
| Queue | `check_queue` |

---

## Reward Scoring

Each episode is scored across 7 dimensions:

| Dimension | Weight | Description |
|---|---|---|
| resolution | 30% | Was a ResolveAction submitted? |
| rca_accuracy | 30% | Root cause correctness (exact / alias / keyword) |
| efficiency | 20% | Steps used vs ideal (exponential decay) |
| tool_economy | 10% | Cumulative tool cost |
| tool_diversity | 5% | Unique tools / total calls |
| safety | 3% | No destructive commands used |
| cmd_quality | 2% | Domain-relevant tools chosen |

Score is multiplied by difficulty (easy 1.0 / medium 1.2 / hard 1.5) and clamped to `[0.01, 0.99]`.

---

## Running Locally

```bash
# Install deps
pip install fastapi uvicorn networkx pydantic openai httpx

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal — run inference (deterministic, no API key needed)
python inference.py

# Run with LLM
API_KEY=your_key MODEL_NAME=gpt-4o-mini python inference.py

# Run baseline (always reproducible)
curl http://localhost:7860/baseline | python -m json.tool
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible LLM endpoint |
| `API_KEY` | *(required for LLM mode)* | HF token or OpenAI key |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `ENV_URL` | `http://localhost:7860` | Server URL |

---

## Log Format

The inference script emits structured logs consumed by the OpenEnv validator:

```
[START] task=dns_failure env=network-diagnostics-env model=gpt-4o-mini
[STEP]  step=1 action=ListToolsAction reward=0.00 done=false error=null
[STEP]  step=2 action=CallToolAction  reward=-0.05 done=false error=null
[STEP]  step=3 action=ResolveAction   reward=0.82 done=true error=null
[GRADER] task=dns_failure score=0.8200 passed=true
[END]   success=true steps=3 score=0.820 rewards=0.00,-0.05,0.82
```
