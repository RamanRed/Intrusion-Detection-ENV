---
title: Network Diagnostics Env
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# NetworkDiagnosticsEnv

A network diagnostics RL environment server. Agents are trained to diagnose and resolve network issues across simulated OS environments.

## API Endpoints

- `GET /health` — Health check
- `POST /reset` — Start a new episode
- `POST /step` — Take an action
- `GET /state` — Get current episode state
- `GET /docs` — Interactive API documentation (Swagger UI)

## Quick Start

```python
import httpx

base = "https://ramanred-my-env.hf.space"

# Reset
r = httpx.post(f"{base}/reset", json={"os_profile": "linux", "difficulty": "medium"})
print(r.json())

# Step
r = httpx.post(f"{base}/step", json={
    "action_type": "ListToolsAction"
})
print(r.json())
```
