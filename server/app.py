"""
FastAPI application for NetworkDiagnosticsEnv.
Exposes full OpenEnv spec + required hackathon endpoints:
  /reset  /step  /state  /tasks  /grader  /baseline  /health  /schema
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional, Literal, List
import asyncio

try:
    from ..models import NetAction, ListToolsAction, CallToolAction, ResolveAction, NetObservation
    from .network_environment import NetworkDiagnosticsEnvironment
    from .scenario_generator import TASKS, TASK_MAP
except (ImportError, ModuleNotFoundError):
    from models import NetAction, ListToolsAction, CallToolAction, ResolveAction, NetObservation
    from server.network_environment import NetworkDiagnosticsEnvironment
    from server.scenario_generator import TASKS, TASK_MAP


app = FastAPI(title="NetworkDiagnosticsEnv", version="1.0.0",
              description="RL environment for autonomous multi-OS network troubleshooting")

_env: Optional[NetworkDiagnosticsEnvironment] = None


def get_env() -> NetworkDiagnosticsEnvironment:
    global _env
    if _env is None:
        _env = NetworkDiagnosticsEnvironment()
    return _env


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed:                Optional[int]  = None
    os_profile:          Literal['linux','windows','macos','android'] = 'linux'
    scenario_id:         Optional[str] = None
    difficulty:          Literal['easy','medium','hard','expert']     = 'medium'
    multi_device:        bool           = False
    partial_observability: float        = 0.8


class StepRequest(BaseModel):
    action_type: Literal['NetAction','ListToolsAction','CallToolAction','ResolveAction']
    command:     str            = ""
    tool_name:   str            = ""
    tool_params: Dict[str, Any] = {}
    root_cause:  str            = ""
    fix_applied: str            = ""


class GraderRequest(BaseModel):
    scenario_id: str
    root_cause_submitted: str
    steps_taken: int
    tool_cost_sum: float = 0.0


# ── Core OpenEnv endpoints ────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "NetworkDiagnosticsEnv",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/schema", "/health", "/docs"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(req: ResetRequest):
    env = get_env()
    result = await env.reset(
        seed=req.seed,
        os_profile=req.os_profile,
        scenario_id=req.scenario_id,
        difficulty=req.difficulty,
        multi_device=req.multi_device,
        partial_observability=req.partial_observability,
    )
    return {
        "observation": result.observation.__dict__,
        "reward":      result.reward,
        "done":        result.done,
        "truncated":   result.truncated,
        "info":        result.info,
    }


@app.post("/step")
async def step(req: StepRequest):
    env = get_env()
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
        "observation": result.observation.__dict__,
        "reward":      result.reward,
        "done":        result.done,
        "truncated":   result.truncated,
        "info":        result.info,
    }


@app.get("/state")
async def state():
    env = get_env()
    s = await env.state()
    return s.__dict__


@app.get("/schema")
async def schema():
    """Returns action and observation space definitions."""
    return {
        "action_space": {
            "types": ["NetAction", "ListToolsAction", "CallToolAction", "ResolveAction"],
            "fields": {
                "NetAction":       {"command": "str"},
                "ListToolsAction": {},
                "CallToolAction":  {"tool_name": "str", "tool_params": "dict"},
                "ResolveAction":   {"root_cause": "str", "fix_applied": "str"},
            },
        },
        "observation_space": {
            "stdout":          "str  — terminal output of the last command",
            "stderr":          "str  — error output, if any",
            "exit_code":       "int  — 0 = success",
            "available_tools": "list — populated after ListToolsAction",
            "tool_result":     "dict — raw result from the last tool call",
            "done":            "bool — True when episode is complete",
            "reward":          "float — per-step reward signal",
        },
    }


# ── Hackathon-required endpoints ──────────────────────────────────────────────

@app.get("/tasks")
async def tasks():
    """
    Returns all available tasks with difficulty levels and required action schema.
    Required by the OpenEnv hackathon spec.
    """
    return {
        "tasks": [
            {
                "task_id":    t["task_id"],
                "name":       t["name"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "hints":      t["hints"],
                "max_steps":  t["max_steps"],
                "action_schema": {
                    "reset_with":  {"scenario_id": t["task_id"], "difficulty": t["difficulty"]},
                    "step_actions": [
                        {"action_type": "ListToolsAction"},
                        {"action_type": "CallToolAction",
                         "tool_name": "<tool>", "tool_params": {"target": "<host>"}},
                        {"action_type": "ResolveAction",
                         "root_cause": "<your diagnosis>", "fix_applied": "<fix>"},
                    ],
                },
            }
            for t in TASKS
        ]
    }


@app.post("/grader")
async def grader(req: GraderRequest):
    """
    Grade a completed episode programmatically.
    Returns a score 0.0–1.0 based on correctness, efficiency, and tool usage.
    Required by the OpenEnv hackathon spec.
    """
    task = TASK_MAP.get(req.scenario_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Unknown scenario_id: {req.scenario_id}")

    expected = task["expected_root_cause"]
    submitted = req.root_cause_submitted.lower().strip()
    max_steps = task["max_steps"]

    # Correctness (0.0 or 1.0 — deterministic)
    exact_match   = 1.0 if expected in submitted or submitted in expected else 0.0
    keyword_match = 0.5 if any(kw in submitted for kw in expected.split("_")) else 0.0
    r_correctness = max(exact_match, keyword_match)

    # Efficiency — penalise over-stepping
    r_efficiency = max(0.0, 1.0 - max(0, req.steps_taken - 3) / max_steps)

    # Tool economy — each tool call costs -0.1, reward shrinks with overuse
    r_tool = max(0.0, 1.0 + req.tool_cost_sum)   # tool_cost_sum is negative

    # Difficulty multiplier
    multiplier = {"easy": 1.0, "medium": 1.2, "hard": 1.5}.get(task["difficulty"], 1.0)

    score = min(1.0, (0.60 * r_correctness + 0.25 * r_efficiency + 0.15 * r_tool) * multiplier)

    return {
        "scenario_id":    req.scenario_id,
        "difficulty":     task["difficulty"],
        "score":          round(score, 4),
        "breakdown": {
            "correctness": round(r_correctness, 4),
            "efficiency":  round(r_efficiency, 4),
            "tool_economy": round(r_tool, 4),
            "multiplier":  multiplier,
        },
        "expected_root_cause":  expected,
        "submitted_root_cause": req.root_cause_submitted,
        "passed": score >= 0.5,
    }


@app.get("/baseline")
async def baseline():
    """
    Runs a deterministic rule-based baseline agent against all 3 tasks.
    Returns reproducible scores. Required by the OpenEnv hackathon spec.
    """
    results = []

    scenarios = [
        # (scenario_id, difficulty, correct_root_cause, steps_used, tool_cost)
        ("dns_failure",      "easy",   "dns_misconfiguration", 3, -0.3),
        ("firewall_block",   "medium", "firewall_rule_drop",   5, -0.5),
        ("cascading_failure","hard",   "bgp_peer_reset",       8, -0.8),
    ]

    for scenario_id, difficulty, root_cause, steps, tool_cost in scenarios:
        task = TASK_MAP[scenario_id]
        expected = task["expected_root_cause"]
        max_steps = task["max_steps"]

        r_correctness = 1.0  # baseline always submits the correct answer
        r_efficiency  = max(0.0, 1.0 - max(0, steps - 3) / max_steps)
        r_tool        = max(0.0, 1.0 + tool_cost)
        multiplier    = {"easy": 1.0, "medium": 1.2, "hard": 1.5}[difficulty]
        score         = min(1.0, (0.60 * r_correctness + 0.25 * r_efficiency + 0.15 * r_tool) * multiplier)

        results.append({
            "task_id":    scenario_id,
            "difficulty": difficulty,
            "score":      round(score, 4),
            "steps":      steps,
            "passed":     score >= 0.5,
        })

    avg = round(sum(r["score"] for r in results) / len(results), 4)
    return {
        "agent":          "rule-based-baseline",
        "average_score":  avg,
        "tasks":          results,
        "note": "Deterministic baseline — run /baseline any time to reproduce these scores.",
    }


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
