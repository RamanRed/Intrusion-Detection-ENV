"""
app.py — NetworkDiagnosticsEnv  (Phase-2 compliant, 3 graders × 3 tasks = 9 tasks)

Grader endpoints:
  POST /grader/connectivity   — dns_failure, dhcp_starvation, firewall_block
  POST /grader/infrastructure — ntp_drift, cascading_failure, routing_loop
  POST /grader/distributed    — split_brain, replica_lag, job_queue_stall

All other OpenEnv required endpoints:
  GET  /health  GET /healthz  POST /reset  POST /step  GET /state
  GET  /tasks   GET /tasks/{id}  GET /baseline  GET /schema
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Literal, List

# Support both package-relative (local dev) and absolute (Docker) imports
try:
    from ..models import NetAction, ListToolsAction, CallToolAction, ResolveAction, NetObservation
    from .network_environment import NetworkDiagnosticsEnvironment
    from .scenario_generator import TASKS, TASK_MAP, TASKS_BY_DOMAIN
    from .reward_engine import RewardEngine
except (ImportError, ModuleNotFoundError):
    from models import NetAction, ListToolsAction, CallToolAction, ResolveAction, NetObservation
    from server.network_environment import NetworkDiagnosticsEnvironment
    from server.scenario_generator import TASKS, TASK_MAP, TASKS_BY_DOMAIN
    from server.reward_engine import RewardEngine


app = FastAPI(
    title="NetworkDiagnosticsEnv",
    version="3.0.0",
    description=(
        "RL environment for autonomous network troubleshooting. "
        "9 tasks across 3 grader domains (connectivity / infrastructure / distributed). "
        "OpenEnv Phase-1 + Phase-2 compliant."
    ),
)

_env: Optional[NetworkDiagnosticsEnvironment] = None


def get_env() -> NetworkDiagnosticsEnvironment:
    global _env
    if _env is None:
        _env = NetworkDiagnosticsEnvironment()
    return _env


# ── Request schemas ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed:                  Optional[int]  = None
    os_profile:            Literal["linux", "windows", "macos", "android"] = "linux"
    scenario_id:           Optional[str]  = None
    difficulty:            Literal["easy", "medium", "hard", "expert"] = "medium"
    multi_device:          bool           = False
    partial_observability: float          = Field(default=0.8, ge=0.0, le=1.0)


class StepRequest(BaseModel):
    action_type: Literal["NetAction", "ListToolsAction", "CallToolAction", "ResolveAction"] = "NetAction"
    command:     str            = ""
    tool_name:   str            = ""
    tool_params: Dict[str, Any] = {}
    root_cause:  str            = ""
    fix_applied: str            = ""


class GraderRequest(BaseModel):
    scenario_id:          str
    root_cause_submitted: str
    steps_taken:          int   = Field(default=1, ge=1)
    tool_cost_sum:        float = 0.0
    tool_names:           List[str] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs_dict(obs) -> Dict[str, Any]:
    d = obs.__dict__.copy()
    for k, v in d.items():
        if hasattr(v, "__dict__"):
            d[k] = v.__dict__
    return d


def _grade(req: GraderRequest, allowed_domain: str) -> Dict[str, Any]:
    """Shared grading logic used by all three /grader/* endpoints."""
    task = TASK_MAP.get(req.scenario_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario_id '{req.scenario_id}'. "
                   f"Valid IDs: {list(TASK_MAP.keys())}",
        )
    if task["domain"] != allowed_domain:
        raise HTTPException(
            status_code=400,
            detail=(
                f"scenario_id '{req.scenario_id}' belongs to domain '{task['domain']}', "
                f"not '{allowed_domain}'. "
                f"Use POST /grader/{task['domain']} instead."
            ),
        )
    engine    = RewardEngine(task["difficulty"])
    breakdown = engine.compute_breakdown(
        target_root_cause=task["expected_root_cause"],
        claimed_cause=req.root_cause_submitted,
        is_resolved=True,
        tools_called=req.steps_taken,
        tool_cost_sum=req.tool_cost_sum,
        tool_names=req.tool_names,
        max_steps=task["max_steps"],
    )
    return {
        "scenario_id":          req.scenario_id,
        "domain":               task["domain"],
        "difficulty":           task["difficulty"],
        "score":                breakdown["total"],
        "passed":               breakdown["passed"],
        "breakdown":            breakdown["dimensions"],
        "multiplier":           breakdown["multiplier"],
        "expected_root_cause":  task["expected_root_cause"],
        "submitted_root_cause": req.root_cause_submitted,
    }


def _task_response(t: Dict[str, Any]) -> Dict[str, Any]:
    domain = t["domain"]
    return {
        "task_id":        t["task_id"],
        "name":           t["name"],
        "domain":         domain,
        "difficulty":     t["difficulty"],
        "description":    t["description"],
        "hints":          t.get("hints", []),
        "max_steps":      t["max_steps"],
        "affected_nodes": t.get("affected_nodes", []),
        "symptom_chain":  t.get("symptom_chain", []),
        "grader": {
            "type":           "api",
            "endpoint":       f"/grader/{domain}",
            "method":         "POST",
            "input_fields":   ["scenario_id", "root_cause_submitted", "steps_taken", "tool_cost_sum"],
            "score_field":    "score",
            "pass_threshold": 0.5,
        },
    }


# ── Meta endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name":    "NetworkDiagnosticsEnv",
        "version": "3.0.0",
        "status":  "running",
        "tasks":   len(TASKS),
        "graders": [
            {"endpoint": "/grader/connectivity",   "tasks": [t["task_id"] for t in TASKS_BY_DOMAIN.get("connectivity", [])]},
            {"endpoint": "/grader/infrastructure", "tasks": [t["task_id"] for t in TASKS_BY_DOMAIN.get("infrastructure", [])]},
            {"endpoint": "/grader/distributed",    "tasks": [t["task_id"] for t in TASKS_BY_DOMAIN.get("distributed", [])]},
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0.0"}


@app.get("/healthz")
async def healthz():
    """Alias for /health — required by OpenEnv validators (matches reference repo pattern)."""
    return {"status": "ok", "version": "3.0.0"}


# ── Core OpenEnv endpoints ────────────────────────────────────────────────────

@app.post("/reset")
async def reset(request: Request):
    """Start/reset an episode. Accepts empty body — all fields have defaults."""
    env = get_env()
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    req    = ResetRequest(**{k: v for k, v in body.items() if v is not None})
    result = await env.reset(
        seed=req.seed,
        os_profile=req.os_profile,
        scenario_id=req.scenario_id,
        difficulty=req.difficulty,
        multi_device=req.multi_device,
        partial_observability=req.partial_observability,
    )
    return {
        "observation": _obs_dict(result.observation),
        "reward":      result.reward,
        "done":        result.done,
        "truncated":   result.truncated,
        "info":        result.info,
    }


@app.post("/step")
async def step(request: Request):
    """Execute one action step."""
    env = get_env()
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    req = StepRequest(**body)

    if req.action_type == "ListToolsAction":
        action = ListToolsAction()
    elif req.action_type == "CallToolAction":
        action = CallToolAction(tool_name=req.tool_name, parameters=req.tool_params)
    elif req.action_type == "ResolveAction":
        action = ResolveAction(root_cause=req.root_cause, fix_applied=req.fix_applied)
    else:
        action = NetAction(command=req.command)

    result = await env.step(action)
    return {
        "observation": _obs_dict(result.observation),
        "reward":      result.reward,
        "done":        result.done,
        "truncated":   result.truncated,
        "info":        result.info,
    }


@app.get("/state")
async def state():
    env = get_env()
    s   = await env.state()
    return s.__dict__


@app.get("/schema")
async def schema():
    return {
        "action_space": {
            "types":  ["NetAction", "ListToolsAction", "CallToolAction", "ResolveAction"],
            "fields": {
                "NetAction":       {"command":    "str"},
                "ListToolsAction": {},
                "CallToolAction":  {"tool_name":  "str", "tool_params": "dict"},
                "ResolveAction":   {"root_cause": "str", "fix_applied": "str"},
            },
        },
        "observation_space": {
            "stdout":          "str",
            "stderr":          "str",
            "exit_code":       "int",
            "available_tools": "list",
            "tool_result":     "dict",
            "done":            "bool",
            "reward":          "float",
        },
    }


# ── Tasks catalogue ───────────────────────────────────────────────────────────

@app.get("/tasks")
async def tasks():
    """
    All 9 tasks grouped by grader domain.
    Each task's 'grader' field points to its domain-specific grader endpoint.
    """
    return {
        "total":   len(TASKS),
        "graders": {
            domain: {
                "endpoint": f"/grader/{domain}",
                "tasks":    [_task_response(t) for t in domain_tasks],
            }
            for domain, domain_tasks in TASKS_BY_DOMAIN.items()
        },
        # Flat list for backwards compatibility
        "tasks": [_task_response(t) for t in TASKS],
    }


@app.get("/tasks/{task_id}")
async def task_detail(task_id: str):
    t = TASK_MAP.get(task_id)
    if not t:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found. Valid IDs: {list(TASK_MAP.keys())}",
        )
    return _task_response(t)


# ══════════════════════════════════════════════════════════════════════════════
#  GRADER 1 — /grader/connectivity
#  Tasks: dns_failure · dhcp_starvation · firewall_block
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/grader/connectivity")
async def grader_connectivity(req: GraderRequest):
    """
    Grade a connectivity-domain episode.
    Accepts: dns_failure | dhcp_starvation | firewall_block
    Returns a deterministic score 0.0–1.0.
    """
    return _grade(req, allowed_domain="connectivity")


# ══════════════════════════════════════════════════════════════════════════════
#  GRADER 2 — /grader/infrastructure
#  Tasks: ntp_drift · cascading_failure · routing_loop
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/grader/infrastructure")
async def grader_infrastructure(req: GraderRequest):
    """
    Grade an infrastructure-domain episode.
    Accepts: ntp_drift | cascading_failure | routing_loop
    Returns a deterministic score 0.0–1.0.
    """
    return _grade(req, allowed_domain="infrastructure")


# ══════════════════════════════════════════════════════════════════════════════
#  GRADER 3 — /grader/distributed
#  Tasks: split_brain · replica_lag · job_queue_stall
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/grader/distributed")
async def grader_distributed(req: GraderRequest):
    """
    Grade a distributed-systems-domain episode.
    Accepts: split_brain | replica_lag | job_queue_stall
    Returns a deterministic score 0.0–1.0.
    """
    return _grade(req, allowed_domain="distributed")


# ── Legacy single /grader endpoint (kept for backwards compatibility) ──────────

@app.post("/grader")
async def grader_legacy(req: GraderRequest):
    """
    Legacy grader — routes to the correct domain grader automatically.
    Prefer /grader/{domain} endpoints for new integrations.
    """
    task = TASK_MAP.get(req.scenario_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario_id '{req.scenario_id}'. "
                   f"Valid IDs: {list(TASK_MAP.keys())}",
        )
    return _grade(req, allowed_domain=task["domain"])


# ── Baseline ──────────────────────────────────────────────────────────────────

@app.get("/baseline")
async def baseline():
    """
    Deterministic rule-based expert baseline across all 9 tasks.
    NO API KEY REQUIRED. Grouped by grader domain.
    """
    # (scenario_id, rca, steps, tool_cost, tool_names)
    expert_runs = [
        # connectivity
        ("dns_failure",      "dns_misconfiguration",                3, -0.15, ["ListToolsAction", "nslookup", "check_logs"]),
        ("dhcp_starvation",  "dhcp_pool_exhausted",                 3, -0.25, ["ListToolsAction", "check_dhcp", "arp_scan"]),
        ("firewall_block",   "firewall_rule_drop",                  4, -0.30, ["ListToolsAction", "ping", "traceroute", "check_iptables"]),
        # infrastructure
        ("ntp_drift",        "ntp_clock_skew",                      4, -0.20, ["ListToolsAction", "curl", "check_ntp", "check_logs"]),
        ("cascading_failure","bgp_peer_reset",                      6, -0.45, ["ListToolsAction", "curl", "check_logs", "traceroute", "check_bgp", "check_logs"]),
        ("routing_loop",     "static_routing_loop",                 4, -0.20, ["ListToolsAction", "ping", "traceroute", "check_logs"]),
        # distributed
        ("split_brain",      "split_brain_misconfigured_heartbeat", 5, -0.35, ["ListToolsAction", "check_cluster", "check_logs", "check_logs", "check_cluster"]),
        ("replica_lag",      "replica_binlog_position_mismatch",    4, -0.20, ["ListToolsAction", "check_logs", "check_service", "check_logs"]),
        ("job_queue_stall",  "worker_crash_missing_env_var",        3, -0.15, ["ListToolsAction", "check_service", "check_logs"]),
    ]

    results_by_domain: Dict[str, List] = {}
    all_results = []

    for scenario_id, rca, steps, tool_cost, tool_names in expert_runs:
        task   = TASK_MAP[scenario_id]
        engine = RewardEngine(task["difficulty"])
        bd     = engine.compute_breakdown(
            target_root_cause=task["expected_root_cause"],
            claimed_cause=rca,
            is_resolved=True,
            tools_called=steps,
            tool_cost_sum=tool_cost,
            tool_names=tool_names,
            max_steps=task["max_steps"],
        )
        entry = {
            "task_id":    scenario_id,
            "domain":     task["domain"],
            "difficulty": task["difficulty"],
            "score":      bd["total"],
            "passed":     bd["passed"],
            "steps":      steps,
        }
        all_results.append(entry)
        results_by_domain.setdefault(task["domain"], []).append(entry)

    avg = round(sum(r["score"] for r in all_results) / len(all_results), 4)

    return {
        "agent":         "rule-based-expert-baseline",
        "version":       "3.0.0",
        "average_score": avg,
        "tasks_total":   len(all_results),
        "tasks_passed":  sum(1 for r in all_results if r["passed"]),
        "by_domain":     results_by_domain,
        "tasks":         all_results,
        "note":          "Deterministic — no API key required. Same result every run.",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse, uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
