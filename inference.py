"""
inference.py — OpenEnv hackathon submission script for NetworkDiagnosticsEnv.

Required env vars:
    API_BASE_URL   The API endpoint for the LLM (OpenAI-compatible)
    MODEL_NAME     The model identifier to use
    HF_TOKEN       Your Hugging Face / API key (no default — must be set)
    ENV_URL        The HF Space URL (default: http://localhost:7860)

Optional env vars:
    LOCAL_IMAGE_NAME   Used when calling from_docker_image()

Logs structured [START], [STEP], [END] lines to stdout as required.
"""

import asyncio
import json
import os
import sys
import time
from typing import List

from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

TASK_NAME        = "NetworkDiagnosticsEnv"
BENCHMARK        = "network-diagnostics-env"
MAX_STEPS        = 20
MAX_TOTAL_REWARD = 3.0          # 3 tasks, each max reward 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

# ── HTTP helpers (no extra deps beyond openai) ────────────────────────────────
try:
    import httpx
    def _post(url: str, body: dict) -> dict:
        with httpx.Client(timeout=30) as c:
            r = c.post(url, json=body); r.raise_for_status(); return r.json()
    def _get(url: str) -> dict:
        with httpx.Client(timeout=30) as c:
            r = c.get(url); r.raise_for_status(); return r.json()
except ImportError:
    import urllib.request
    def _post(url: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req  = urllib.request.Request(url, data=data,
               headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    def _get(url: str) -> dict:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())

# ── Structured logging (required format) ─────────────────────────────────────

def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(*, step: int, action: str, reward: float, done: bool, error) -> None:
    err_str = f" error={error}" if error is not None else ""
    print(f"[STEP] step={step} reward={reward} done={done}{err_str}", flush=True)

def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] task={TASK_NAME} score={score:.4f} steps={steps} success={success}", flush=True)

def log_grader(*, task: str, score: float, passed: bool) -> None:
    print(f"[GRADER] task={task} score={score:.4f} passed={passed}", flush=True)


SYSTEM_PROMPT = (
    "You are a network SRE agent diagnosing a simulated network failure.\n"
    "Reply with ONLY a single JSON action object — no prose, no markdown.\n"
    "Available action types:\n"
    '  {"action_type":"ListToolsAction"}\n'
    '  {"action_type":"CallToolAction","tool_name":"<n>","tool_params":{"target":"<host>"}}\n'
    '  {"action_type":"ResolveAction","root_cause":"<rca>","fix_applied":"<fix>"}\n'
    "Diagnose efficiently. When confident, submit ResolveAction."
)


def get_model_action(client: OpenAI, messages: list) -> dict:
    """Call the LLM and return a parsed action dict. Falls back to ListToolsAction on error."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if model wraps in them
        raw = raw.strip("```json").strip("```").strip()
        return json.loads(raw), raw
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "ListToolsAction"}, '{"action_type":"ListToolsAction"}'


# ── Single-task episode ───────────────────────────────────────────────────────

def run_task_episode(client: OpenAI, task: dict) -> dict:
    scenario_id = task["task_id"]
    difficulty  = task["difficulty"]
    max_steps   = task.get("max_steps", MAX_STEPS)

    rewards: List[float] = []
    steps_taken = 0

    # Reset
    obs_resp = _post(f"{ENV_URL}/reset", {
        "scenario_id": scenario_id,
        "difficulty":  difficulty,
        "os_profile":  "linux",
    })
    initial_obs = obs_resp["observation"]["stdout"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Scenario: {task['description']}\n"
            f"Initial observation: {initial_obs}\n"
            "Begin. Start with ListToolsAction."
        )},
    ]

    done = False
    for step in range(1, max_steps + 1):
        if done:
            break

        action_dict, action_raw = get_model_action(client, messages)
        messages.append({"role": "assistant", "content": action_raw})

        result  = _post(f"{ENV_URL}/step", action_dict)
        obs_    = result["observation"]
        reward  = result.get("reward") or 0.0
        done    = result.get("done", False)
        out     = obs_.get("stdout") or obs_.get("stderr") or ""

        rewards.append(reward)
        steps_taken = step

        log_step(step=step, action=action_raw, reward=reward, done=done, error=None)
        messages.append({"role": "user", "content": f"Observation: {out[:300]}"})

        if done:
            break

    score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
    score = min(max(score, 0.0), 1.0)

    # ── Call /grader for official per-task score ──────────────────────────────
    tool_cost_sum = sum(r for r in rewards if r < 0)  # negative rewards are tool penalties
    try:
        grader_resp = _post(f"{ENV_URL}/grader", {
            "scenario_id":          scenario_id,
            "root_cause_submitted": "",   # best-effort; LLM may have resolved correctly
            "steps_taken":          steps_taken,
            "tool_cost_sum":        round(tool_cost_sum, 4),
        })
        grader_score  = grader_resp.get("score", score)
        grader_passed = grader_resp.get("passed", grader_score >= SUCCESS_SCORE_THRESHOLD)
    except Exception as exc:
        print(f"[DEBUG] Grader call failed for {scenario_id}: {exc}", flush=True)
        grader_score  = score
        grader_passed = score >= SUCCESS_SCORE_THRESHOLD

    log_grader(task=scenario_id, score=grader_score, passed=grader_passed)

    return {"score": grader_score, "rewards": rewards, "steps": steps_taken, "passed": grader_passed}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    all_rewards: List[float] = []
    all_steps   = 0
    success     = False

    try:
        # Fetch tasks from the running environment
        tasks_resp = _get(f"{ENV_URL}/tasks")
        tasks      = tasks_resp["tasks"]

        for task in tasks:
            print(f"[DEBUG] Running task: {task['task_id']}", flush=True)
            result = run_task_episode(client, task)
            all_rewards.extend(result["rewards"])
            all_steps  += result["steps"]

        total_score = sum(all_rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        total_score = min(max(total_score, 0.0), 1.0)
        success     = total_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Fatal error: {exc}", flush=True)
        total_score = 0.0

    finally:
        log_end(success=success, steps=all_steps, score=total_score, rewards=all_rewards)


if __name__ == "__main__":
    main()
