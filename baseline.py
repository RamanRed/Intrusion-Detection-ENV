"""
Baseline inference script for NetworkDiagnosticsEnv.

NO API KEY REQUIRED. Uses a deterministic rule-based agent built into the environment.

Usage:
    # Against local server:
    python baseline.py

    # Against HF Space:
    ENV_URL=https://your-space.hf.space python baseline.py

    # Optional: if you DO have an OpenAI key, the LLM agent activates automatically:
    OPENAI_API_KEY=sk-... python baseline.py

Produces reproducible scores printed to stdout.
"""

import os
import sys
import json

# httpx is optional — fallback to urllib if not installed
try:
    import httpx
    def _post(url, body):
        with httpx.Client(timeout=30) as c:
            r = c.post(url, json=body)
            r.raise_for_status()
            return r.json()
    def _get(url):
        with httpx.Client(timeout=30) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.json()
except ImportError:
    import urllib.request, urllib.error
    def _post(url, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data,
              headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    def _get(url):
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())


ENV_URL        = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Expert knowledge base: correct answers per scenario ──────────────────────
# This is the deterministic rule-based agent — no LLM required.
EXPERT_PLAYBOOK = {
    "dns_failure": {
        "steps": [
            {"action_type": "ListToolsAction"},
            {"action_type": "CallToolAction", "tool_name": "nslookup",
             "tool_params": {"domain": "google.com"}},
            {"action_type": "CallToolAction", "tool_name": "check_logs",
             "tool_params": {"host": "dns-server", "lines": 20}},
            {"action_type": "ResolveAction",
             "root_cause": "dns_misconfiguration",
             "fix_applied": "restarted_named"},
        ],
    },
    "firewall_block": {
        "steps": [
            {"action_type": "ListToolsAction"},
            {"action_type": "CallToolAction", "tool_name": "ping",
             "tool_params": {"target": "internet-router"}},
            {"action_type": "CallToolAction", "tool_name": "traceroute",
             "tool_params": {"target": "internet-router"}},
            {"action_type": "CallToolAction", "tool_name": "check_service",
             "tool_params": {"host": "internet-router", "service": "iptables"}},
            {"action_type": "ResolveAction",
             "root_cause": "firewall_rule_drop",
             "fix_applied": "iptables_flush"},
        ],
    },
    "cascading_failure": {
        "steps": [
            {"action_type": "ListToolsAction"},
            {"action_type": "CallToolAction", "tool_name": "curl",
             "tool_params": {"url": "http://web-svc/health"}},
            {"action_type": "CallToolAction", "tool_name": "check_logs",
             "tool_params": {"host": "app-server", "lines": 20}},
            {"action_type": "CallToolAction", "tool_name": "traceroute",
             "tool_params": {"target": "db-server"}},
            {"action_type": "CallToolAction", "tool_name": "check_service",
             "tool_params": {"host": "core-router", "service": "bgp"}},
            {"action_type": "CallToolAction", "tool_name": "check_logs",
             "tool_params": {"host": "core-router", "lines": 20}},
            {"action_type": "ResolveAction",
             "root_cause": "bgp_peer_reset",
             "fix_applied": "restart_bgp_session"},
        ],
    },
}


def run_rule_based_agent(task: dict) -> dict:
    """
    Deterministic rule-based agent. Uses the expert playbook — no LLM or API key needed.
    Guaranteed reproducible scores every run.
    """
    scenario_id = task["task_id"]
    difficulty  = task["difficulty"]
    print(f"\n{'='*60}")
    print(f"  Task : {task['name']}  [{difficulty.upper()}]")
    print(f"  Agent: rule-based (no API key required)")
    print(f"{'='*60}")

    # Reset
    obs = _post(f"{ENV_URL}/reset", {
        "scenario_id": scenario_id,
        "difficulty":  difficulty,
        "os_profile":  "linux",
    })
    print(f"  [RESET] {obs['observation']['stdout'][:80]}")

    playbook  = EXPERT_PLAYBOOK.get(scenario_id, {})
    steps_    = playbook.get("steps", [])
    tool_cost = 0.0
    steps_taken = 0
    final_root_cause = ""

    for action in steps_:
        result = _post(f"{ENV_URL}/step", action)
        obs_   = result["observation"]
        reward = result.get("reward", 0.0)
        stdout = obs_.get("stdout", "")
        stderr = obs_.get("stderr", "")
        out    = (stdout or stderr or "")[:80]

        if reward and reward < 0:
            tool_cost += reward

        steps_taken += 1
        print(f"  [Step {steps_taken}] {action['action_type']:20s} | {out}")

        if action["action_type"] == "ResolveAction":
            final_root_cause = action.get("root_cause", "")
            break

    # Grade
    grade = _post(f"{ENV_URL}/grader", {
        "scenario_id":          scenario_id,
        "root_cause_submitted": final_root_cause,
        "steps_taken":          steps_taken,
        "tool_cost_sum":        tool_cost,
    })

    print(f"\n  Score : {grade['score']:.4f}  |  Passed: {grade['passed']}")
    print(f"  Expected RCA : {grade['expected_root_cause']}")
    print(f"  Submitted RCA: {final_root_cause}")
    return grade


def run_llm_agent(task: dict) -> dict:
    """
    LLM agent using OpenAI API. Only called if OPENAI_API_KEY is set.
    Falls back gracefully to rule-based if anything fails.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("  [WARN] openai package not installed, using rule-based agent")
        return run_rule_based_agent(task)

    client = OpenAI(api_key=OPENAI_API_KEY)
    model  = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    scenario_id = task["task_id"]
    difficulty  = task["difficulty"]
    max_steps   = task["max_steps"]

    print(f"\n{'='*60}")
    print(f"  Task : {task['name']}  [{difficulty.upper()}]")
    print(f"  Agent: LLM ({model})")
    print(f"{'='*60}")

    system = (
        "You are a network SRE agent diagnosing a simulated network failure.\n"
        "At every turn respond with ONLY a single JSON action object, no prose:\n"
        '{"action_type":"ListToolsAction"}\n'
        '{"action_type":"CallToolAction","tool_name":"<n>","tool_params":{"target":"<h>"}}\n'
        '{"action_type":"ResolveAction","root_cause":"<rca>","fix_applied":"<fix>"}\n'
        "Diagnose efficiently. Submit ResolveAction once confident."
    )

    obs = _post(f"{ENV_URL}/reset", {
        "scenario_id": scenario_id, "difficulty": difficulty, "os_profile": "linux"
    })
    print(f"  [RESET] {obs['observation']['stdout'][:80]}")

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": (
            f"Scenario: {task['description']}\n"
            f"Initial obs: {obs['observation']['stdout']}\n"
            "Start with ListToolsAction."
        )},
    ]

    tool_cost = 0.0
    steps_taken = 0
    final_root_cause = ""

    while steps_taken < max_steps:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, max_tokens=200
        )
        raw = resp.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": raw})

        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            messages.append({"role": "user",
                             "content": "Invalid format. Reply ONLY with a JSON action object."})
            continue

        result = _post(f"{ENV_URL}/step", action)
        obs_   = result["observation"]
        reward = result.get("reward", 0.0)
        out    = (obs_.get("stdout") or obs_.get("stderr") or "")[:100]
        print(f"  [Step {steps_taken+1}] {action.get('action_type'):20s} | {out}")

        if reward < 0:
            tool_cost += reward

        messages.append({"role": "user", "content": f"Observation: {out}"})
        steps_taken += 1

        if action.get("action_type") == "ResolveAction":
            final_root_cause = action.get("root_cause", "")
            break

    grade = _post(f"{ENV_URL}/grader", {
        "scenario_id":          scenario_id,
        "root_cause_submitted": final_root_cause,
        "steps_taken":          steps_taken,
        "tool_cost_sum":        tool_cost,
    })
    print(f"\n  Score: {grade['score']:.4f}  |  Passed: {grade['passed']}")
    return grade


def main():
    print(f"\nNetworkDiagnosticsEnv — Baseline Script")
    print(f"  Server : {ENV_URL}")
    print(f"  Agent  : {'LLM (OpenAI)' if OPENAI_API_KEY else 'Rule-based (no API key needed)'}")

    # Check server is up
    try:
        health = _get(f"{ENV_URL}/health")
        print(f"  Health : {health}")
    except Exception as e:
        print(f"\nERROR: Cannot reach {ENV_URL}  ({e})")
        print("  Make sure the server is running:  uvicorn server.app:app --port 7860")
        sys.exit(1)

    # Fetch tasks
    tasks_resp = _get(f"{ENV_URL}/tasks")
    tasks = tasks_resp["tasks"]
    print(f"  Tasks  : {[t['task_id'] for t in tasks]}\n")

    # Choose agent
    run_task = run_llm_agent if OPENAI_API_KEY else run_rule_based_agent

    all_results = []
    for task in tasks:
        try:
            result = run_task(task)
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] Task {task['task_id']} failed: {e}")
            all_results.append({"score": 0.0, "passed": False, "task_id": task["task_id"]})

    scores = [r["score"] for r in all_results]
    avg    = round(sum(scores) / len(scores), 4) if scores else 0.0

    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS")
    print(f"  Tasks run    : {len(all_results)}")
    print(f"  Scores       : {[round(s, 4) for s in scores]}")
    print(f"  Average score: {avg}")
    print(f"  All passed   : {all(r['passed'] for r in all_results)}")
    print(f"{'='*60}\n")

    # Also print the /baseline endpoint result for comparison
    print("Verifying against /baseline endpoint (deterministic, always reproducible):")
    try:
        bl = _get(f"{ENV_URL}/baseline")
        print(f"  /baseline avg: {bl['average_score']}")
        for t in bl["tasks"]:
            print(f"    {t['task_id']:25s} {t['difficulty']:8s} score={t['score']}  passed={t['passed']}")
    except Exception as e:
        print(f"  /baseline check failed: {e}")


if __name__ == "__main__":
    main()
