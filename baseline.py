"""
Baseline inference script for NetworkDiagnosticsEnv.
Uses OpenAI API client to run an LLM agent against all 3 tasks.

Usage:
    export OPENAI_API_KEY=sk-...
    export ENV_URL=https://ramanred-my-env.hf.space   # or http://localhost:7860
    python baseline.py

Produces reproducible scores printed to stdout.
"""

import os
import json
import asyncio
import httpx
from openai import OpenAI

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a network troubleshooting agent. You are given access to a simulated network environment.
Your goal is to diagnose the root cause of a network failure by calling tools, then submit a ResolveAction.

At each step, respond ONLY with a JSON object in one of these formats:
{"action_type": "ListToolsAction"}
{"action_type": "CallToolAction", "tool_name": "<name>", "tool_params": {"target": "<host>"}}
{"action_type": "ResolveAction", "root_cause": "<your diagnosis>", "fix_applied": "<fix>"}

Be concise. Diagnose efficiently. Submit ResolveAction as soon as you are confident.
"""


def call_env(path: str, method: str = "GET", body: dict = None) -> dict:
    with httpx.Client(timeout=30) as client:
        if method == "POST":
            r = client.post(f"{ENV_URL}{path}", json=body)
        else:
            r = client.get(f"{ENV_URL}{path}")
        r.raise_for_status()
        return r.json()


def run_task(client: OpenAI, task: dict) -> dict:
    scenario_id = task["task_id"]
    difficulty   = task["difficulty"]
    max_steps    = task["max_steps"]

    print(f"\n{'='*60}")
    print(f"Task: {task['name']} [{difficulty}]")
    print(f"{'='*60}")

    # Reset environment
    obs_result = call_env("/reset", "POST", {
        "scenario_id": scenario_id,
        "difficulty":  difficulty,
        "os_profile":  "linux",
    })
    print(f"[RESET] {obs_result['observation']['stdout']}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Scenario: {task['description']}\n"
            f"Initial observation: {obs_result['observation']['stdout']}\n"
            "Begin diagnosis. Use ListToolsAction first."
        )},
    ]

    steps = 0
    tool_cost = 0.0
    final_root_cause = ""
    done = False

    while steps < max_steps and not done:
        # Get LLM action
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": raw})

        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  [Step {steps}] LLM returned non-JSON, retrying...")
            messages.append({"role": "user", "content": "Invalid format. Respond ONLY with a JSON action object."})
            continue

        print(f"  [Step {steps}] Action: {action.get('action_type')} {action.get('tool_name','')}")

        # Execute action in env
        step_result = call_env("/step", "POST", action)
        obs = step_result["observation"]
        reward = step_result["reward"]
        done = step_result["done"]
        tool_cost += reward if reward < 0 else 0

        stdout = obs.get("stdout", "")
        stderr = obs.get("stderr", "")
        output = stdout or stderr or str(obs.get("tool_result", ""))
        print(f"  [Step {steps}] Obs: {output[:120]}")

        messages.append({"role": "user", "content": f"Observation: {output}"})

        if action.get("action_type") == "ResolveAction":
            final_root_cause = action.get("root_cause", "")
            done = True

        steps += 1

    # Grade via /grader
    grade = call_env("/grader", "POST", {
        "scenario_id": scenario_id,
        "root_cause_submitted": final_root_cause,
        "steps_taken": steps,
        "tool_cost_sum": tool_cost,
    })

    print(f"\n  Score: {grade['score']} | Passed: {grade['passed']}")
    print(f"  Expected: {grade['expected_root_cause']} | Got: {final_root_cause}")
    return grade


def main():
    if not OPENAI_API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Fetch task list
    tasks_resp = call_env("/tasks")
    tasks = tasks_resp["tasks"]

    all_scores = []
    for task in tasks:
        result = run_task(client, task)
        all_scores.append(result["score"])

    avg = round(sum(all_scores) / len(all_scores), 4)
    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS")
    print(f"  Tasks run:     {len(all_scores)}")
    print(f"  Scores:        {all_scores}")
    print(f"  Average score: {avg}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
