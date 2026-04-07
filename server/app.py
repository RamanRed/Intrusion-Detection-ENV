"""
FastAPI application for the NetworkDiagnosticsEnv Environment.
Inlined server bootstrap — no openenv_core dependency needed in Docker.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional, Literal, Union
import asyncio

try:
    from ..models import NetAction, ListToolsAction, CallToolAction, ResolveAction, NetObservation
    from .network_environment import NetworkDiagnosticsEnvironment
except (ImportError, ModuleNotFoundError):
    from models import NetAction, ListToolsAction, CallToolAction, ResolveAction, NetObservation
    from server.network_environment import NetworkDiagnosticsEnvironment


app = FastAPI(title="NetworkDiagnosticsEnv", version="1.0.0")

# Single shared environment instance (max_concurrent_envs=1)
_env: Optional[NetworkDiagnosticsEnvironment] = None


def get_env() -> NetworkDiagnosticsEnvironment:
    global _env
    if _env is None:
        _env = NetworkDiagnosticsEnvironment()
    return _env


# ── Pydantic request/response models ────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    os_profile: Literal['linux', 'windows', 'macos', 'android'] = 'linux'
    scenario_id: Optional[str] = None
    difficulty: Literal['easy', 'medium', 'hard', 'expert'] = 'medium'
    multi_device: bool = False
    partial_observability: float = 0.8


class StepRequest(BaseModel):
    action_type: Literal['NetAction', 'ListToolsAction', 'CallToolAction', 'ResolveAction']
    # NetAction
    command: str = ""
    # CallToolAction
    tool_name: str = ""
    tool_params: Dict[str, Any] = {}
    # ResolveAction
    root_cause: str = ""
    fix_applied: str = ""


# ── Routes ────────────────────────────────────────────────────────────────────

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
        "reward": result.reward,
        "done": result.done,
        "truncated": result.truncated,
        "info": result.info,
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
    else:  # NetAction
        action = NetAction(command=req.command)

    result = await env.step(action)
    return {
        "observation": result.observation.__dict__,
        "reward": result.reward,
        "done": result.done,
        "truncated": result.truncated,
        "info": result.info,
    }


@app.get("/state")
async def state():
    env = get_env()
    s = await env.state()
    return s.__dict__


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
