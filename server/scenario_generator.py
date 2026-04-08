import networkx as nx
from typing import Dict, Any, Tuple


# Task registry — used by /tasks endpoint
TASKS = [
    {
        "task_id": "dns_failure",
        "name": "DNS Server Failure",
        "difficulty": "easy",
        "description": (
            "The DNS server has crashed due to a misconfiguration in named.conf. "
            "Hosts can no longer resolve domain names. "
            "Identify the root cause and submit a ResolveAction."
        ),
        "hints": ["Try pinging the DNS server", "Use nslookup to verify resolution failure"],
        "expected_root_cause": "dns_misconfiguration",
        "max_steps": 10,
        "grader": {
            "type": "api",
            "endpoint": "/grader",
            "method": "POST",
            "input_fields": ["scenario_id", "root_cause_submitted", "steps_taken", "tool_cost_sum"],
            "score_field": "score",
            "pass_threshold": 0.5,
        },
    },
    {
        "task_id": "firewall_block",
        "name": "Firewall Blocking Outbound Traffic",
        "difficulty": "medium",
        "description": (
            "An iptables rule on the internet-router is silently dropping all outbound packets "
            "from host-a. Internal traffic works; only internet access is broken. "
            "Identify the firewall rule causing the drop and submit a ResolveAction."
        ),
        "hints": ["Internal ping works, external does not", "Check iptables rules on the router"],
        "expected_root_cause": "firewall_rule_drop",
        "max_steps": 15,
        "grader": {
            "type": "api",
            "endpoint": "/grader",
            "method": "POST",
            "input_fields": ["scenario_id", "root_cause_submitted", "steps_taken", "tool_cost_sum"],
            "score_field": "score",
            "pass_threshold": 0.5,
        },
    },
    {
        "task_id": "cascading_failure",
        "name": "Cascading Multi-Hop Service Failure",
        "difficulty": "hard",
        "description": (
            "A web server (web-svc) is returning 502 errors. The root cause is a chain: "
            "the backend database lost its network route due to a BGP peer reset on core-router, "
            "which caused the app-server's DB connection pool to exhaust, "
            "which caused web-svc to return 502. "
            "Trace the full dependency chain and identify the true root cause node."
        ),
        "hints": [
            "Start from the symptom (web-svc 502), trace upstream",
            "Check BGP sessions on core-router",
            "DB connection pool errors are a symptom, not the cause",
        ],
        "expected_root_cause": "bgp_peer_reset",
        "max_steps": 25,
        "grader": {
            "type": "api",
            "endpoint": "/grader",
            "method": "POST",
            "input_fields": ["scenario_id", "root_cause_submitted", "steps_taken", "tool_cost_sum"],
            "score_field": "score",
            "pass_threshold": 0.5,
        },
    },
]

TASK_MAP = {t["task_id"]: t for t in TASKS}


class ScenarioGenerator:
    def __init__(self):
        pass

    def generate(self, scenario_id: str, os_profile: str, difficulty: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        graph = nx.DiGraph()

        # ── Base topology ────────────────────────────────────────────────────
        graph.add_node("host-a",          os=os_profile,  ip="10.0.0.5",   status="up")
        graph.add_node("dns-server",      os="linux",     ip="10.0.0.1",   status="up")
        graph.add_node("internet-router", os="linux",     ip="10.0.0.254", status="up")

        graph.add_edge("host-a", "dns-server",      latency_ms=1)
        graph.add_edge("host-a", "internet-router", latency_ms=2)

        ground_truth = {
            "scenario_id":      scenario_id,
            "root_cause_node":  "",
            "root_cause":       "",
            "fix_applied":      "",
        }

        # ── Scenario-specific faults ─────────────────────────────────────────
        if scenario_id == "dns_failure":
            graph.nodes["dns-server"]["status"] = "down"
            ground_truth.update({
                "root_cause_node": "dns-server",
                "root_cause":      "dns_misconfiguration",
                "fix_applied":     "restarted_named",
            })

        elif scenario_id == "firewall_block":
            graph.edges[("host-a", "internet-router")]["firewall_blocked"] = True
            ground_truth.update({
                "root_cause_node": "internet-router",
                "root_cause":      "firewall_rule_drop",
                "fix_applied":     "iptables_flush",
            })

        elif scenario_id == "cascading_failure":
            # Extended topology for the hard scenario
            graph.add_node("web-svc",     os="linux", ip="10.1.0.10", status="up",  http_status=502)
            graph.add_node("app-server",  os="linux", ip="10.1.0.20", status="up",  db_pool_exhausted=True)
            graph.add_node("db-server",   os="linux", ip="10.1.0.30", status="up",  reachable=False)
            graph.add_node("core-router", os="linux", ip="10.1.0.1",  status="up",  bgp_session="down")

            graph.add_edge("web-svc",    "app-server",  latency_ms=2)
            graph.add_edge("app-server", "db-server",   latency_ms=1,  connection_refused=True)
            graph.add_edge("db-server",  "core-router", latency_ms=1,  route_lost=True)
            graph.add_edge("host-a",     "web-svc",     latency_ms=5)

            ground_truth.update({
                "root_cause_node": "core-router",
                "root_cause":      "bgp_peer_reset",
                "fix_applied":     "restart_bgp_session",
            })

        return graph, ground_truth
