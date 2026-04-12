---
title: NetworkDiagnosticsEnv
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: OpenEnv RL env for network troubleshooting
tags:
  - openenv
  - reinforcement-learning
  - network-diagnostics
  - fastapi
---

# NetworkDiagnosticsEnv

An **OpenEnv-compliant** reinforcement-learning environment for evaluating AI agents on autonomous network troubleshooting. Phase 1 and Phase 2 compliant — 3 grader domains x 3 tasks each = 9 tasks total.

## Grader Domains

| Grader endpoint | Tasks |
|---|---|
| POST /grader/connectivity | dns_failure, dhcp_starvation, firewall_block |
| POST /grader/infrastructure | ntp_drift, cascading_failure, routing_loop |
| POST /grader/distributed | split_brain, replica_lag, job_queue_stall |

## Running Locally

```bash
pip install fastapi uvicorn networkx pydantic openai httpx
uvicorn server.app:app --host 0.0.0.0 --port 7860
python inference.py
```

## Environment Variables

- API_BASE_URL — OpenAI-compatible LLM endpoint
- API_KEY — HF token or OpenAI key
- MODEL_NAME — Model identifier (default: gpt-4o-mini)
- ENV_URL — Server URL (default: http://localhost:7860)
