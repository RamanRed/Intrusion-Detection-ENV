"""
reward_engine.py — Multi-dimensional reward engine for NetworkDiagnosticsEnv.

Scoring dimensions (weights sum to 1.0):
  resolution    0.30  — was the issue actually resolved?
  rca_accuracy  0.30  — how accurate is the root-cause diagnosis?
  efficiency    0.20  — steps used vs ideal (smooth exponential decay)
  tool_economy  0.10  — cumulative tool cost penalty
  tool_diversity 0.05 — penalise repetitive tool usage
  safety        0.03  — no destructive commands detected
  cmd_quality   0.02  — domain-relevant tools used

Difficulty multipliers: easy 1.0 / medium 1.2 / hard 1.5 / expert 2.0
All scores clamped to (0.01, 0.99) — never exactly 0 or 1.
"""

import math
from typing import Dict, Any, List


class RewardEngine:

    IDEAL_STEPS: Dict[str, int] = {
        "easy":   3,
        "medium": 5,
        "hard":   8,
        "expert": 12,
    }

    DIFFICULTY_MULTIPLIER: Dict[str, float] = {
        "easy":   1.0,
        "medium": 1.2,
        "hard":   1.5,
        "expert": 2.0,
    }

    WEIGHTS: Dict[str, float] = {
        "resolution":     0.30,
        "rca_accuracy":   0.30,
        "efficiency":     0.20,
        "tool_economy":   0.10,
        "tool_diversity": 0.05,
        "safety":         0.03,
        "cmd_quality":    0.02,
    }

    # Domain keyword → relevant tool names
    DOMAIN_TOOLS: Dict[str, set] = {
        "dns":        {"nslookup", "check_logs", "check_service"},
        "dhcp":       {"check_dhcp", "arp_scan", "check_logs"},
        "firewall":   {"check_iptables", "ping", "traceroute"},
        "ntp":        {"check_ntp", "check_logs", "check_service", "curl"},
        "bgp":        {"check_bgp", "traceroute", "check_logs", "curl"},
        "routing":    {"check_routes", "traceroute", "ping", "check_logs"},
        "cluster":    {"check_cluster", "check_logs"},
        "replica":    {"check_replica", "check_logs", "check_service"},
        "worker":     {"check_service", "check_logs", "check_queue"},
        "split":      {"check_cluster", "check_logs"},
        "heartbeat":  {"check_cluster", "check_logs"},
        "binlog":     {"check_replica", "check_logs"},
        "env":        {"check_service", "check_logs", "check_queue"},
        "pool":       {"check_dhcp", "arp_scan", "check_logs"},
    }

    def __init__(self, difficulty: str = "medium"):
        self.difficulty  = difficulty
        self.multiplier  = self.DIFFICULTY_MULTIPLIER.get(difficulty, 1.0)
        self.ideal_steps = self.IDEAL_STEPS.get(difficulty, 5)

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, target_root_cause: str, claimed_cause: str, is_resolved: bool,
                tools_called: int, tool_cost_sum: float,
                tool_names: List[str] = None, max_steps: int = None) -> float:
        return self.compute_breakdown(
            target_root_cause=target_root_cause, claimed_cause=claimed_cause,
            is_resolved=is_resolved, tools_called=tools_called,
            tool_cost_sum=tool_cost_sum, tool_names=tool_names or [],
            max_steps=max_steps or self.ideal_steps * 3,
        )["total"]

    def compute_breakdown(self, target_root_cause: str, claimed_cause: str,
                          is_resolved: bool, tools_called: int, tool_cost_sum: float,
                          tool_names: List[str] = None, max_steps: int = None) -> Dict[str, Any]:
        tool_names = tool_names or []
        max_steps  = max_steps  or self.ideal_steps * 3

        r_resolution  = self._score_resolution(is_resolved)
        r_rca         = self._score_rca(target_root_cause, claimed_cause)
        r_efficiency  = self._score_efficiency(tools_called, max_steps)
        r_economy     = self._score_tool_economy(tool_cost_sum)
        r_diversity   = self._score_tool_diversity(tool_names)
        r_safety      = self._score_safety(tool_names)
        r_cmd_quality = self._score_cmd_quality(tool_names, target_root_cause)

        raw = (
            self.WEIGHTS["resolution"]     * r_resolution  +
            self.WEIGHTS["rca_accuracy"]   * r_rca         +
            self.WEIGHTS["efficiency"]     * r_efficiency   +
            self.WEIGHTS["tool_economy"]   * r_economy      +
            self.WEIGHTS["tool_diversity"] * r_diversity    +
            self.WEIGHTS["safety"]         * r_safety       +
            self.WEIGHTS["cmd_quality"]    * r_cmd_quality
        ) * self.multiplier

        total = round(min(0.99, max(0.01, raw)), 4)

        return {
            "total":      total,
            "difficulty": self.difficulty,
            "multiplier": self.multiplier,
            "passed":     total >= 0.5,
            "dimensions": {
                "resolution":     round(r_resolution,  4),
                "rca_accuracy":   round(r_rca,         4),
                "efficiency":     round(r_efficiency,   4),
                "tool_economy":   round(r_economy,      4),
                "tool_diversity": round(r_diversity,    4),
                "safety":         round(r_safety,       4),
                "cmd_quality":    round(r_cmd_quality,  4),
            },
            "weights": self.WEIGHTS,
        }

    def compute_penalty_violation(self) -> float:
        return -10.0

    # ── Dimension scorers ─────────────────────────────────────────────────────

    def _score_resolution(self, is_resolved: bool) -> float:
        return 1.0 if is_resolved else 0.0

    def _score_rca(self, target: str, claimed: str) -> float:
        """
        Semantic root-cause matching with three tiers:
          1.0 — exact substring match (case-insensitive)
          0.7 — synonym / alias match
          0.5 — ≥60% keyword overlap
          0.3 — any keyword overlap
          0.0 — no overlap
        """
        t = target.lower().strip().replace("-", "_")
        c = claimed.lower().strip().replace("-", "_")

        if not c:
            return 0.0
        if t == c or t in c or c in t:
            return 1.0

        # Synonym aliases for common phrasings agents use
        aliases: Dict[str, List[str]] = {
            "dns_misconfiguration":                ["named_config", "bind_config", "dns_config_error", "named_misconfiguration"],
            "dhcp_pool_exhausted":                 ["dhcp_exhausted", "dhcp_starvation", "ip_exhaustion", "pool_exhaustion"],
            "firewall_rule_drop":                  ["iptables_drop", "firewall_drop", "packet_drop", "iptables_block"],
            "ntp_clock_skew":                      ["clock_drift", "ntp_drift", "time_drift", "clock_skew"],
            "bgp_peer_reset":                      ["bgp_down", "bgp_session_down", "bgp_reset", "bgp_failure"],
            "static_routing_loop":                 ["routing_loop", "route_loop", "loop_route", "static_loop"],
            "split_brain_misconfigured_heartbeat": ["split_brain", "dual_leader", "heartbeat_misconfiguration", "raft_split"],
            "replica_binlog_position_mismatch":    ["binlog_mismatch", "replica_lag", "replication_error", "slave_error"],
            "worker_crash_missing_env_var":        ["missing_env", "env_var_missing", "worker_crash", "crash_loop"],
        }
        for canonical, alias_list in aliases.items():
            if canonical == t:
                for alias in alias_list:
                    if alias in c or c in alias:
                        return 0.7

        # Keyword overlap
        t_kws = set(t.split("_")) - {""}
        c_kws = set(c.split("_")) - {""}
        if not t_kws:
            return 0.0
        overlap = len(t_kws & c_kws) / len(t_kws)
        if overlap >= 0.6:
            return 0.5
        if overlap > 0.0:
            return 0.3
        return 0.0

    def _score_efficiency(self, steps_taken: int, max_steps: int) -> float:
        """
        Reward agents that solve quickly.
        Score = 1.0 at or below ideal, exponential decay above ideal.
        Minimum 0.05 to avoid zeroing the overall score.
        """
        if steps_taken <= self.ideal_steps:
            return 1.0
        oversteps   = steps_taken - self.ideal_steps
        decay_range = max(1, max_steps - self.ideal_steps)
        score       = math.exp(-2.5 * oversteps / decay_range)
        return max(0.05, round(score, 4))

    def _score_tool_economy(self, tool_cost_sum: float) -> float:
        """Map cumulative (negative) tool cost to [0.05, 1.0]."""
        return max(0.05, min(1.0, 1.0 + tool_cost_sum))

    def _score_tool_diversity(self, tool_names: List[str]) -> float:
        """
        Penalise repeating the same tool.
        unique / total — floor 0.05.
        """
        if not tool_names:
            return 1.0
        unique = len(set(tool_names))
        total  = len(tool_names)
        return max(0.05, round(unique / total, 4))

    def _score_safety(self, tool_names: List[str]) -> float:
        """Penalise use of destructive-sounding commands."""
        destructive = {"rm", "dd", "mkfs", "format", "shutdown", "reboot", "kill", "drop_table"}
        for name in tool_names:
            if any(d in name.lower() for d in destructive):
                return 0.0
        return 1.0

    def _score_cmd_quality(self, tool_names: List[str], target_root_cause: str) -> float:
        """
        Reward using tools that are relevant to the root cause domain.
        Maps domain keywords in the RCA string to expected tool sets.
        """
        if not tool_names:
            return 0.5
        used = set(tool_names)
        best = 0.0
        rca  = target_root_cause.lower()
        for keyword, domain_tools in self.DOMAIN_TOOLS.items():
            if keyword in rca:
                hits      = len(used & domain_tools)
                hit_ratio = hits / max(len(domain_tools), 1)
                best      = max(best, hit_ratio)
        return round(max(0.3, best), 4) if best > 0 else 0.5
