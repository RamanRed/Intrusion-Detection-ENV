"""
baseline.py — Advanced deterministic + LLM baseline for NetworkDiagnosticsEnv.

NO API KEY REQUIRED for the rule-based agent.
All 6 tasks are covered with an expert playbook.

Usage:
    # Rule-based (no API key):
    python baseline.py

    # LLM agent:
    OPENAI_API_KEY=sk-... python baseline.py

    # Against HF Space:
    ENV_URL=https://your-space.hf.space python baseline.py
"""

import json
import os
import sys

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
    import urllib.request
    def _post(url, body):
        data = json.dumps(body).encode()
        req  = urllib.request.Request(url, data=data,
               headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    def _get(url):
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())


ENV_URL        = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# ── Expert playbook for rule-based agent ──────────────────────────────────────
# Each entry is a list of action dicts sent in sequence.
# The expert always uses the minimal correct tool path.

EXPERT_PLAYBOOK = {
    # ── Easy ──────────────────────────────────────────────────────────────────
    "dns_failure": {
        "steps": [
            {"action_type": "ListToolsAction"},
            {"action_type": "CallToolAction", "tool_name": "nslookup",
             "tool_params": {"domain": "google.com"}},
            {"action_type": "CallToolAction", "tool_name": "check_logs",
             "tool_params": {"host": "dns-server", "lines": 20}},
            {"action_type": "ResolveAction",
             "root_cause": "dns_misconfiguration",
             "fix_applied": "fixed_named_conf_and_restarted_named"},
        ],
    },
    "dhcp_starvation": {
        "steps": [
            {"action_type": "ListToolsAction"},
            {"action_type": "CallToolAction", "tool_name": "check_dhcp",
             "tool_params": {"host": "dhcp-server"}},
            {"action_type": "CallToolAction", "tool_name": "arp_scan",
             "tool_params": {"subnet": "10.0.0.0/24"}},
            {"action_type": "ResolveAction",
             "root_cause": "dhcp_pool_exhausted",
             "fix_applied": "blocked_rogue_mac_flushed_stale_leases"},
        ],
    },

    # ── Medium ────────────────────────────────────────────────────────────────
    "firewall_block": {
        "steps": [
            {"action_type": "ListToolsAction"},
            {"action_type": "CallToolAction", "tool_name": "ping",
             "tool_params": {"target": "internet-router"}},
            {"action_type": "CallToolAction", "tool_name": "traceroute",
             "tool_params": {"target": "internet-router"}},
            {"action_type": "CallToolAction", "tool_name": "check_iptables",
             "tool_params": {"host": "internet-router", "chain": "FORWARD"}},
            {"action_type": "ResolveAction",
             "root_cause": "firewall_rule_drop",
             "fix_applied": "iptables_delete_drop_rule"},
        ],
    },
    "ntp_drift": {
        "steps": [
            {"action_type": "ListToolsAction"},
            {"action_type": "CallToolAction", "tool_name": "curl",
             "tool_params": {"url": "https://internal-svc/health"}},
            {"action_type": "CallToolAction", "tool_name": "check_ntp",
             "tool_params": {"host": "host-a"}},
            {"action_type": "CallToolAction", "tool_name": "check_service",
             "tool_params": {"host": "ntp-server", "service": "ntpd"}},
            {"action_type": "ResolveAction",
             "root_cause": "ntp_clock_skew",
             "fix_applied": "restarted_ntpd_and_force_synced_clocks"},
        ],
    },

    # ── Hard ──────────────────────────────────────────────────────────────────
    "cascading_failure": {
        "steps": [
            {"action_type": "ListToolsAction"},
            {"action_type": "CallToolAction", "tool_name": "curl",
             "tool_params": {"url": "http://web-svc/health"}},
            {"action_type": "CallToolAction", "tool_name": "check_logs",
             "tool_params": {"host": "app-server", "lines": 30}},
            {"action_type": "CallToolAction", "tool_name": "traceroute",
             "tool_params": {"target": "db-server"}},
            {"action_type": "CallToolAction", "tool_name": "check_bgp",
             "tool_params": {"host": "core-router"}},
            {"action_type": "CallToolAction", "tool_name": "check_logs",
             "tool_params": {"host": "core-router", "lines": 30}},
            {"action_type": "ResolveAction",
             "root_cause": "bgp_peer_reset",
             "fix_applied": "restart_bgp_session_on_core_router"},
        ],
    },
    "split_brain": {
        "steps": [
            {"action_type": "ListToolsAction"},
            {"action_type": "CallToolAction", "tool_name": "check_cluster",
             "tool_params": {"host": "cluster-node-1"}},
            {"action_type": "CallToolAction", "tool_name": "check_logs",
             "tool_params": {"host": "cluster-node-2", "lines": 30}},
            {"action_type": "CallToolAction", "tool_name": "check_logs",
             "tool_params": {"host": "cluster-node-1", "lines": 30}},
            {"action_type": "CallToolAction", "tool_name": "check_cluster",
             "tool_params": {"host": "cluster-node-2"}},
            {"action_type": "ResolveAction",
             "root_cause": "split_brain_misconfigured_heartbeat",
             "fix_applied": "fenced_stale_leader_increased_heartbeat_timeout"},
        ],
    },
}


# ── Rule-based agent ──────────────────────────────────────────────────────────

def run_rule_based_agent(task: dict) -> dict:
    scenario_id = task["task_id"]
    difficulty  = task["difficulty"]

    print(f"\n{'='*65}")
    print(f"  Task      : {task['name']}  [{difficulty.upper()}]")
    print(f"  Agent     : rule-based expert (no API key required)")
    print(f"  Scenario  : {scenario_id}")
    print(f"{'='*65}")

    obs = _post(f"{ENV_URL}/reset", {
        "scenario_id": scenario_id,
        "difficulty":  difficulty,
        "os_profile":  "linux",
    })
    print(f"  [RESET] {obs['observation']['stdout'][:100]}")

    playbook     = EXPERT_PLAYBOOK.get(scenario_id, {})
    steps_       = playbook.get("steps", [])
    tool_cost    = 0.0
    tool_names   = []
    steps_taken  = 0
    final_rca    = ""

    for action in steps_:
        result = _post(f"{ENV_URL}/step", action)
        obs_   = result["observation"]
        reward = result.get("reward", 0.0)
        out    = (obs_.get("stdout") or obs_.get("stderr") or "")[:100]
        meta   = obs_.get("info", {}).get("tool_metadata", {})

        if reward < 0:
            tool_cost += reward

        atype = action["action_type"]
        if atype == "CallToolAction":
            tool_names.append(action.get("tool_name", "?"))

        steps_taken += 1
        meta_str = f"  metadata={json.dumps(meta)}" if meta else ""
        print(f"  [Step {steps_taken:2d}] {atype:25s} | {out}{meta_str}")

        if atype == "ResolveAction":
            final_rca = action.get("root_cause", "")
            break

    grade = _post(f"{ENV_URL}/grader", {
        "scenario_id":          scenario_id,
        "root_cause_submitted": final_rca,
        "steps_taken":          steps_taken,
        "tool_cost_sum":        tool_cost,
        "tool_names":           tool_names,
    })

    print(f"\n  Score   : {grade['score']:.4f}  |  Passed: {grade['passed']}")
    print(f"  Expected: {grade['expected_root_cause']}")
    print(f"  Got     : {final_rca}")
    if "breakdown" in grade:
        for dim, val in grade["breakdown"].items():
            print(f"    {dim:18s}: {val:.4f}")

    return grade


# ── LLM agent ─────────────────────────────────────────────────────────────────

def run_llm_agent(task: dict) -> dict:
    try:
        from openai import OpenAI
    except ImportError:
        print("  [WARN] openai package not installed — falling back to rule-based agent")
        return run_rule_based_agent(task)

    client      = OpenAI(api_key=OPENAI_API_KEY)
    model       = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    scenario_id = task["task_id"]
    difficulty  = task["difficulty"]
    max_steps   = task["max_steps"]

    print(f"\n{'='*65}")
    print(f"  Task  : {task['name']}  [{difficulty.upper()}]")
    print(f"  Agent : LLM ({model})")
    print(f"{'='*65}")

    system = (
        "You are a network SRE agent. At each turn reply ONLY with a single JSON action:\n"
        '{"action_type":"ListToolsAction"}\n'
        '{"action_type":"CallToolAction","tool_name":"<n>","tool_params":{...}}\n'
        '{"action_type":"ResolveAction","root_cause":"<rca>","fix_applied":"<fix>"}\n'
        "Known root-cause strings: dns_misconfiguration | dhcp_pool_exhausted | "
        "firewall_rule_drop | ntp_clock_skew | bgp_peer_reset | split_brain_misconfigured_heartbeat"
    )

    obs = _post(f"{ENV_URL}/reset", {
        "scenario_id": scenario_id, "difficulty": difficulty, "os_profile": "linux"
    })
    print(f"  [RESET] {obs['observation']['stdout'][:100]}")

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": (
            f"Scenario: {task.get('description', task['name'])}\n"
            f"Initial obs: {obs['observation']['stdout']}\n"
            "Start with ListToolsAction."
        )},
    ]

    tool_cost   = 0.0
    tool_names  = []
    steps_taken = 0
    final_rca   = ""

    while steps_taken < max_steps:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, max_tokens=300
        )
        raw = resp.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": raw})

        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            messages.append({"role": "user", "content": "Reply ONLY with a JSON action object."})
            continue

        result = _post(f"{ENV_URL}/step", action)
        obs_   = result["observation"]
        reward = result.get("reward", 0.0)
        out    = (obs_.get("stdout") or obs_.get("stderr") or "")[:120]
        meta   = obs_.get("info", {}).get("tool_metadata", {})

        if reward < 0:
            tool_cost += reward
        atype = action.get("action_type", "?")
        if atype == "CallToolAction":
            tool_names.append(action.get("tool_name", "?"))

        steps_taken += 1
        print(f"  [Step {steps_taken:2d}] {atype:25s} | {out}")

        feedback = out
        if meta:
            feedback += f"\n[metadata] {json.dumps(meta)}"
        messages.append({"role": "user", "content": f"Observation: {feedback}"})

        if atype == "ResolveAction":
            final_rca = action.get("root_cause", "")
            break

    grade = _post(f"{ENV_URL}/grader", {
        "scenario_id":          scenario_id,
        "root_cause_submitted": final_rca,
        "steps_taken":          steps_taken,
        "tool_cost_sum":        tool_cost,
        "tool_names":           tool_names,
    })
    print(f"\n  Score: {grade['score']:.4f}  |  Passed: {grade['passed']}")
    return grade


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    agent_label = f"LLM ({os.environ.get('OPENAI_MODEL','gpt-4o-mini')})" if OPENAI_API_KEY else "Rule-based (no API key)"
    print(f"\nNetworkDiagnosticsEnv v2.0 — Baseline Script")
    print(f"  Server : {ENV_URL}")
    print(f"  Agent  : {agent_label}")

    try:
        health = _get(f"{ENV_URL}/health")
        print(f"  Health : {health}")
    except Exception as e:
        print(f"\nERROR: Cannot reach {ENV_URL}  ({e})")
        print("  Start the server: uvicorn server.app:app --port 7860")
        sys.exit(1)

    tasks_resp = _get(f"{ENV_URL}/tasks")
    tasks      = tasks_resp["tasks"]
    print(f"  Tasks  : {[t['task_id'] for t in tasks]}  ({len(tasks)} total)\n")

    run_task    = run_llm_agent if OPENAI_API_KEY else run_rule_based_agent
    all_results = []

    for task in tasks:
        try:
            result = run_task(task)
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] Task {task['task_id']} failed: {e}")
            all_results.append({"score": 0.0, "passed": False})

    scores  = [r["score"] for r in all_results]
    avg     = round(sum(scores) / len(scores), 4) if scores else 0.0
    n_pass  = sum(1 for r in all_results if r["passed"])

    print(f"\n{'='*65}")
    print(f"BASELINE RESULTS")
    print(f"  Tasks run    : {len(all_results)}")
    print(f"  Tasks passed : {n_pass} / {len(all_results)}")
    print(f"  Scores       : {[round(s, 4) for s in scores]}")
    print(f"  Average score: {avg}")
    print(f"{'='*65}\n")

    print("Verifying against /baseline endpoint (deterministic):")
    try:
        bl = _get(f"{ENV_URL}/baseline")
        print(f"  /baseline avg  : {bl['average_score']}")
        print(f"  /baseline tasks: {bl['tasks_total']}  passed={bl['tasks_passed']}")
        for t in bl["tasks"]:
            print(f"    {t['task_id']:35s} [{t['difficulty']:6s}] "
                  f"score={t['score']:.4f}  passed={t['passed']}")
    except Exception as e:
        print(f"  /baseline check failed: {e}")


if __name__ == "__main__":
    main()
