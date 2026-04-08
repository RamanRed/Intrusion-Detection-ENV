"""
scenario_generator.py — Advanced scenario registry and topology generator.

Improvements over v1:
  - 6 tasks (was 3): 2 easy / 2 medium / 2 hard
  - Every task carries a grader descriptor (required by Phase-2 validator)
  - Richer ground-truth metadata: affected_nodes, symptom_chain, fix_steps
  - ScenarioGenerator produces realistic partial-observability noise
  - Difficulty gates: partial_observability degrades visibility on hard tasks
"""

import random
import networkx as nx
from typing import Dict, Any, Tuple, List, Optional


# ── Grader descriptor reused by every task ────────────────────────────────────
def _grader_spec(pass_threshold: float = 0.5) -> Dict[str, Any]:
    return {
        "type":         "api",
        "endpoint":     "/grader",
        "method":       "POST",
        "input_fields": ["scenario_id", "root_cause_submitted", "steps_taken", "tool_cost_sum"],
        "score_field":  "score",
        "pass_threshold": pass_threshold,
    }


# ── Task registry ─────────────────────────────────────────────────────────────
TASKS: List[Dict[str, Any]] = [

    # ── Easy ─────────────────────────────────────────────────────────────────
    {
        "task_id":   "dns_failure",
        "name":      "DNS Server Failure",
        "difficulty": "easy",
        "description": (
            "The DNS server has crashed due to a misconfiguration in /etc/named.conf. "
            "Hosts can no longer resolve domain names. "
            "Identify the root cause and submit a ResolveAction."
        ),
        "hints": [
            "Try nslookup to confirm DNS is unreachable",
            "Check logs on dns-server for named startup errors",
        ],
        "expected_root_cause":  "dns_misconfiguration",
        "affected_nodes":       ["dns-server"],
        "symptom_chain":        ["nslookup fails", "named.conf parse error", "named process down"],
        "fix_steps":            ["fix /etc/named.conf", "systemctl restart named"],
        "max_steps":            10,
        "grader":               _grader_spec(0.5),
    },
    {
        "task_id":   "dhcp_starvation",
        "name":      "DHCP Pool Exhaustion",
        "difficulty": "easy",
        "description": (
            "New hosts are failing to obtain IP addresses. The DHCP server has run out of "
            "available leases due to a rogue device flooding DHCP requests. "
            "Identify the root cause and submit a ResolveAction."
        ),
        "hints": [
            "Check the DHCP lease table on dhcp-server",
            "Look for unusually high lease request rates in logs",
        ],
        "expected_root_cause":  "dhcp_pool_exhausted",
        "affected_nodes":       ["dhcp-server"],
        "symptom_chain":        ["new hosts get no IP", "DHCP lease table full", "rogue device flooding"],
        "fix_steps":            ["block rogue MAC", "flush stale leases", "expand pool range"],
        "max_steps":            10,
        "grader":               _grader_spec(0.5),
    },

    # ── Medium ────────────────────────────────────────────────────────────────
    {
        "task_id":   "firewall_block",
        "name":      "Firewall Blocking Outbound Traffic",
        "difficulty": "medium",
        "description": (
            "An iptables rule on the internet-router is silently dropping all outbound packets "
            "from host-a. Internal traffic works fine; only internet access is broken. "
            "Identify the firewall rule causing the drop and submit a ResolveAction."
        ),
        "hints": [
            "Internal ping works, external does not — narrow it to the gateway",
            "Check iptables OUTPUT/FORWARD chains on internet-router",
        ],
        "expected_root_cause":  "firewall_rule_drop",
        "affected_nodes":       ["internet-router"],
        "symptom_chain":        ["external ping fails", "iptables DROP rule on FORWARD chain"],
        "fix_steps":            ["iptables -D FORWARD -s 10.0.0.5 -j DROP", "save rules"],
        "max_steps":            15,
        "grader":               _grader_spec(0.5),
    },
    {
        "task_id":   "ntp_drift",
        "name":      "NTP Clock Skew Breaking TLS",
        "difficulty": "medium",
        "description": (
            "Internal HTTPS services are failing with certificate validation errors. "
            "The ntp-server process crashed and host clocks have drifted by >5 minutes. "
            "TLS handshakes reject certificates as 'not yet valid' or 'expired'. "
            "Identify the root cause and submit a ResolveAction."
        ),
        "hints": [
            "Check date/time on affected hosts — large skew breaks TLS",
            "Look for ntpd/chronyd process status on ntp-server",
        ],
        "expected_root_cause":  "ntp_clock_skew",
        "affected_nodes":       ["ntp-server"],
        "symptom_chain":        ["TLS cert errors", "clock drift >5 min", "ntpd crashed"],
        "fix_steps":            ["systemctl restart ntpd", "ntpdate -u pool.ntp.org"],
        "max_steps":            15,
        "grader":               _grader_spec(0.5),
    },

    # ── Hard ──────────────────────────────────────────────────────────────────
    {
        "task_id":   "cascading_failure",
        "name":      "Cascading Multi-Hop Service Failure",
        "difficulty": "hard",
        "description": (
            "A web server (web-svc) is returning 502 errors. The root cause is a chain: "
            "the backend database lost its route due to a BGP peer reset on core-router, "
            "which caused the app-server DB connection pool to exhaust, "
            "which caused web-svc to return 502. "
            "Trace the full dependency chain and identify the true root cause node."
        ),
        "hints": [
            "Start from the symptom (web-svc 502) and trace upstream",
            "Check BGP sessions on core-router — BGP is the likely root cause",
            "DB pool exhaustion is a symptom, not the root cause",
        ],
        "expected_root_cause":  "bgp_peer_reset",
        "affected_nodes":       ["core-router", "db-server", "app-server", "web-svc"],
        "symptom_chain":        ["web-svc 502", "app-server pool exhausted", "db unreachable", "BGP down on core-router"],
        "fix_steps":            ["restart BGP session on core-router", "wait for route reconvergence"],
        "max_steps":            25,
        "grader":               _grader_spec(0.5),
    },
    {
        "task_id":   "split_brain",
        "name":      "Split-Brain Cluster with Stale Leader",
        "difficulty": "hard",
        "description": (
            "A distributed key-value cluster (3 nodes) has two nodes claiming leadership "
            "after a network partition. Writes are being accepted by both leaders, causing "
            "data divergence. A misconfigured heartbeat timeout caused the follower to "
            "incorrectly trigger a leader election during a transient blip. "
            "Identify the root cause and submit a ResolveAction."
        ),
        "hints": [
            "Check cluster membership logs on all 3 nodes",
            "Look for split-brain quorum errors and heartbeat timeout config",
            "The follower triggered an election too early — compare timeout values",
        ],
        "expected_root_cause":  "split_brain_misconfigured_heartbeat",
        "affected_nodes":       ["cluster-node-1", "cluster-node-2", "cluster-node-3"],
        "symptom_chain":        ["dual leader", "write divergence", "premature election", "heartbeat timeout too low"],
        "fix_steps":            ["fence stale leader", "increase heartbeat timeout", "trigger re-election", "resync data"],
        "max_steps":            30,
        "grader":               _grader_spec(0.5),
    },
]

TASK_MAP: Dict[str, Dict[str, Any]] = {t["task_id"]: t for t in TASKS}


# ── Scenario generator ────────────────────────────────────────────────────────

class ScenarioGenerator:
    """
    Builds a networkx topology graph + ground-truth metadata for a given scenario.
    Supports partial-observability noise injection for hard scenarios.
    """

    def __init__(self):
        self._rng = random.Random()

    def generate(
        self,
        scenario_id: str,
        os_profile: str,
        difficulty: str,
        seed: Optional[int] = None,
        partial_observability: float = 1.0,
    ) -> Tuple[nx.DiGraph, Dict[str, Any]]:

        if seed is not None:
            self._rng.seed(seed)

        graph = nx.DiGraph()
        ground_truth: Dict[str, Any] = {
            "scenario_id":      scenario_id,
            "root_cause_node":  "",
            "root_cause":       "",
            "fix_applied":      "",
            "affected_nodes":   [],
            "symptom_chain":    [],
        }

        builder = getattr(self, f"_build_{scenario_id}", self._build_default)
        builder(graph, ground_truth, os_profile, difficulty, partial_observability)
        return graph, ground_truth

    # ── Base topology shared by all scenarios ─────────────────────────────────
    def _add_base_topology(
        self,
        graph: nx.DiGraph,
        os_profile: str,
        partial_observability: float,
    ) -> None:
        """Add core nodes. Under partial observability some attributes are hidden."""
        nodes = [
            ("host-a",          os_profile,   "10.0.0.5",   "up"),
            ("dns-server",      "linux",       "10.0.0.1",   "up"),
            ("internet-router", "linux",       "10.0.0.254", "up"),
            ("dhcp-server",     "linux",       "10.0.0.2",   "up"),
            ("ntp-server",      "linux",       "10.0.0.3",   "up"),
        ]
        for name, os_, ip, status in nodes:
            attrs: Dict[str, Any] = {"os": os_, "ip": ip, "status": status}
            # Partial observability: randomly mask status on hard scenarios
            if partial_observability < 0.8 and self._rng.random() > partial_observability:
                attrs["status"] = "unknown"
            graph.add_node(name, **attrs)

        graph.add_edge("host-a", "dns-server",      latency_ms=1)
        graph.add_edge("host-a", "internet-router", latency_ms=2)
        graph.add_edge("host-a", "dhcp-server",     latency_ms=1)
        graph.add_edge("host-a", "ntp-server",      latency_ms=1)

    # ── Scenario builders ──────────────────────────────────────────────────────

    def _build_dns_failure(self, graph, gt, os_profile, difficulty, po):
        self._add_base_topology(graph, os_profile, po)
        graph.nodes["dns-server"]["status"]          = "down"
        graph.nodes["dns-server"]["named_exit_code"] = 1
        graph.nodes["dns-server"]["named_error"]     = "unknown option 'recursion-limit' in /etc/named.conf:14"
        gt.update({
            "root_cause_node": "dns-server",
            "root_cause":      "dns_misconfiguration",
            "fix_applied":     "restarted_named",
            "affected_nodes":  ["dns-server"],
            "symptom_chain":   ["nslookup fails", "named.conf parse error", "named process down"],
        })

    def _build_dhcp_starvation(self, graph, gt, os_profile, difficulty, po):
        self._add_base_topology(graph, os_profile, po)
        graph.nodes["dhcp-server"]["pool_exhausted"]    = True
        graph.nodes["dhcp-server"]["leases_total"]      = 254
        graph.nodes["dhcp-server"]["leases_used"]       = 254
        graph.nodes["dhcp-server"]["rogue_mac"]         = "de:ad:be:ef:00:01"
        graph.nodes["dhcp-server"]["rogue_req_rate"]    = 1200  # req/min
        gt.update({
            "root_cause_node": "dhcp-server",
            "root_cause":      "dhcp_pool_exhausted",
            "fix_applied":     "blocked_rogue_mac_flushed_leases",
            "affected_nodes":  ["dhcp-server"],
            "symptom_chain":   ["new hosts get no IP", "lease table full", "rogue device flooding"],
        })

    def _build_firewall_block(self, graph, gt, os_profile, difficulty, po):
        self._add_base_topology(graph, os_profile, po)
        graph.edges[("host-a", "internet-router")]["firewall_blocked"] = True
        graph.edges[("host-a", "internet-router")]["iptables_rule"]    = "-A FORWARD -s 10.0.0.5 -j DROP"
        gt.update({
            "root_cause_node": "internet-router",
            "root_cause":      "firewall_rule_drop",
            "fix_applied":     "iptables_flush",
            "affected_nodes":  ["internet-router"],
            "symptom_chain":   ["external ping fails", "iptables DROP rule on FORWARD chain"],
        })

    def _build_ntp_drift(self, graph, gt, os_profile, difficulty, po):
        self._add_base_topology(graph, os_profile, po)
        graph.nodes["ntp-server"]["status"]       = "down"
        graph.nodes["ntp-server"]["clock_drift_s"] = 380   # ~6 min drift
        graph.nodes["ntp-server"]["ntpd_error"]   = "ntpd: segmentation fault (core dumped)"
        # All other hosts show TLS errors as a symptom
        for n in ["host-a", "dns-server", "internet-router"]:
            graph.nodes[n]["tls_errors"] = True
            graph.nodes[n]["clock_drift_s"] = 380
        gt.update({
            "root_cause_node": "ntp-server",
            "root_cause":      "ntp_clock_skew",
            "fix_applied":     "restarted_ntpd_and_synced_clocks",
            "affected_nodes":  ["ntp-server", "host-a", "dns-server"],
            "symptom_chain":   ["TLS cert errors", "clock drift >5 min", "ntpd crashed"],
        })

    def _build_cascading_failure(self, graph, gt, os_profile, difficulty, po):
        self._add_base_topology(graph, os_profile, po)
        # Extended topology
        extra = [
            ("web-svc",         "linux", "10.1.0.10", "up"),
            ("app-server",      "linux", "10.1.0.20", "up"),
            ("db-server",       "linux", "10.1.0.30", "up"),
            ("core-router",     "linux", "10.1.0.1",  "up"),
        ]
        for name, os_, ip, status in extra:
            graph.add_node(name, os=os_, ip=ip, status=status)

        graph.nodes["web-svc"]["http_status"]          = 502
        graph.nodes["app-server"]["db_pool_exhausted"] = True
        graph.nodes["app-server"]["pool_size"]         = 100
        graph.nodes["app-server"]["pool_used"]         = 100
        graph.nodes["db-server"]["reachable"]          = False
        graph.nodes["core-router"]["bgp_session"]      = "down"
        graph.nodes["core-router"]["bgp_peer"]         = "10.1.0.254"
        graph.nodes["core-router"]["bgp_reset_reason"] = "hold_timer_expired"

        graph.add_edge("web-svc",    "app-server",  latency_ms=2)
        graph.add_edge("app-server", "db-server",   latency_ms=1,  connection_refused=True)
        graph.add_edge("db-server",  "core-router", latency_ms=1,  route_lost=True)
        graph.add_edge("host-a",     "web-svc",     latency_ms=5)

        gt.update({
            "root_cause_node": "core-router",
            "root_cause":      "bgp_peer_reset",
            "fix_applied":     "restart_bgp_session",
            "affected_nodes":  ["core-router", "db-server", "app-server", "web-svc"],
            "symptom_chain":   ["web-svc 502", "app-server pool exhausted", "db unreachable", "BGP down on core-router"],
        })

    def _build_split_brain(self, graph, gt, os_profile, difficulty, po):
        self._add_base_topology(graph, os_profile, po)
        cluster_nodes = [
            ("cluster-node-1", "linux", "10.2.0.1", "up", "leader",   300),
            ("cluster-node-2", "linux", "10.2.0.2", "up", "leader",   100),   # stale leader
            ("cluster-node-3", "linux", "10.2.0.3", "up", "follower", 300),
        ]
        for name, os_, ip, status, role, hb_timeout_ms in cluster_nodes:
            graph.add_node(
                name, os=os_, ip=ip, status=status,
                cluster_role=role,
                heartbeat_timeout_ms=hb_timeout_ms,
                split_brain=(role == "leader" and name == "cluster-node-2"),
            )

        # Partial partition: node-2 ↔ node-1 link was flaky
        graph.add_edge("cluster-node-1", "cluster-node-2", latency_ms=2, flaky=True)
        graph.add_edge("cluster-node-1", "cluster-node-3", latency_ms=1)
        graph.add_edge("cluster-node-2", "cluster-node-3", latency_ms=1)
        graph.add_edge("cluster-node-3", "cluster-node-1", latency_ms=1)

        gt.update({
            "root_cause_node": "cluster-node-2",
            "root_cause":      "split_brain_misconfigured_heartbeat",
            "fix_applied":     "fenced_stale_leader_increased_timeout",
            "affected_nodes":  ["cluster-node-1", "cluster-node-2", "cluster-node-3"],
            "symptom_chain":   ["dual leader", "write divergence", "premature election", "heartbeat timeout too low"],
        })

    def _build_default(self, graph, gt, os_profile, difficulty, po):
        """Fallback for unknown scenario IDs."""
        self._add_base_topology(graph, os_profile, po)
        gt.update({"root_cause": "unknown", "root_cause_node": "", "fix_applied": ""})
