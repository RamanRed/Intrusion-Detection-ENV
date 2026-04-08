"""
reward_engine.py — Advanced multi-dimensional reward engine.

Improvements over v1:
  - 7 reward dimensions (was 6)
  - Path-efficiency scoring: penalises revisiting the same tool
  - Semantic RCA matching: partial keyword overlap, not just substring
  - Structured breakdown returned for transparency
  - Difficulty multipliers calibrated to ideal step targets
  - All scores normalised to (0.0, 1.0) before combining
"""

from typing import Dict, Any, List, Tuple
import math


class RewardEngine:

    # Ideal step counts per difficulty level
    IDEAL_STEPS: Dict[str, int] = {
        "easy":   3,
        "medium": 5,
        "hard":   8,
        "expert": 12,
    }

    # Difficulty multipliers applied to the combined weighted score
    DIFFICULTY_MULTIPLIER: Dict[str, float] = {
        "easy":   1.0,
        "medium": 1.2,
        "hard":   1.5,
        "expert": 2.0,
    }

    # Weight for each reward dimension (must sum to 1.0)
    WEIGHTS: Dict[str, float] = {
        "resolution":   0.30,  # Was the issue actually resolved?
        "rca_accuracy": 0.30,  # How accurate is the root-cause diagnosis?
        "efficiency":   0.20,  # Steps used vs ideal
        "tool_economy": 0.10,  # Financial/cost penalty of tools used
        "tool_diversity": 0.05,  # Penalise redundant repeated tool calls
        "safety":       0.03,  # No destructive commands run
        "cmd_quality":  0.02,  # Estimated quality of commands chosen
    }

    def __init__(self, difficulty: str = "medium"):
        self.difficulty        = difficulty
        self.multiplier        = self.DIFFICULTY_MULTIPLIER.get(difficulty, 1.0)
        self.ideal_steps       = self.IDEAL_STEPS.get(difficulty, 5)
        self._tool_call_log: List[str] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(
        self,
        target_root_cause: str,
        claimed_cause: str,
        is_resolved: bool,
        tools_called: int,
        tool_cost_sum: float,
        tool_names: List[str] = None,
        max_steps: int = None,
    ) -> float:
        """
        Compute a scalar reward in [0.0, 1.0] (before difficulty multiplier)
        then scale by multiplier. Returned value is clamped to (0.01, 0.99).
        """
        breakdown = self.compute_breakdown(
            target_root_cause=target_root_cause,
            claimed_cause=claimed_cause,
            is_resolved=is_resolved,
            tools_called=tools_called,
            tool_cost_sum=tool_cost_sum,
            tool_names=tool_names or [],
            max_steps=max_steps or (self.ideal_steps * 3),
        )
        return breakdown["total"]

    def compute_breakdown(
        self,
        target_root_cause: str,
        claimed_cause: str,
        is_resolved: bool,
        tools_called: int,
        tool_cost_sum: float,
        tool_names: List[str] = None,
        max_steps: int = None,
    ) -> Dict[str, Any]:
        """
        Full reward computation returning per-dimension scores and the weighted total.
        """
        tool_names = tool_names or []
        max_steps  = max_steps or (self.ideal_steps * 3)

        # ── Per-dimension scores ──────────────────────────────────────────────
        r_resolution   = self._score_resolution(is_resolved)
        r_rca          = self._score_rca(target_root_cause, claimed_cause)
        r_efficiency   = self._score_efficiency(tools_called, max_steps)
        r_tool_economy = self._score_tool_economy(tool_cost_sum)
        r_diversity    = self._score_tool_diversity(tool_names)
        r_safety       = 1.0   # extended: could analyse tool_names for `rm -rf` etc.
        r_cmd_quality  = self._score_cmd_quality(tool_names, target_root_cause)

        # ── Weighted combination ──────────────────────────────────────────────
        raw = (
            self.WEIGHTS["resolution"]     * r_resolution   +
            self.WEIGHTS["rca_accuracy"]   * r_rca          +
            self.WEIGHTS["efficiency"]     * r_efficiency    +
            self.WEIGHTS["tool_economy"]   * r_tool_economy  +
            self.WEIGHTS["tool_diversity"] * r_diversity     +
            self.WEIGHTS["safety"]         * r_safety        +
            self.WEIGHTS["cmd_quality"]    * r_cmd_quality
        ) * self.multiplier

        total = round(min(0.99, max(0.01, raw)), 4)

        return {
            "total":             total,
            "difficulty":        self.difficulty,
            "multiplier":        self.multiplier,
            "dimensions": {
                "resolution":    round(r_resolution,   4),
                "rca_accuracy":  round(r_rca,          4),
                "efficiency":    round(r_efficiency,    4),
                "tool_economy":  round(r_tool_economy,  4),
                "tool_diversity":round(r_diversity,     4),
                "safety":        round(r_safety,        4),
                "cmd_quality":   round(r_cmd_quality,   4),
            },
            "weights":           self.WEIGHTS,
            "passed":            total >= 0.5,
        }

    def compute_penalty_violation(self) -> float:
        """Hard penalty for a safety / policy violation."""
        return -10.0

    # ── Dimension scorers ──────────────────────────────────────────────────────

    def _score_resolution(self, is_resolved: bool) -> float:
        return 1.0 if is_resolved else 0.0

    def _score_rca(self, target: str, claimed: str) -> float:
        """
        Semantic root-cause matching:
          1.0 — exact substring match
          0.6 — >50% keyword overlap
          0.3 — any keyword overlap
          0.0 — no overlap
        """
        t = target.lower().strip()
        c = claimed.lower().strip()

        if not c:
            return 0.0
        if t in c or c in t:
            return 1.0

        t_keywords = set(t.replace("-", "_").split("_"))
        c_keywords = set(c.replace("-", "_").split("_"))
        t_keywords.discard("")
        c_keywords.discard("")

        if not t_keywords:
            return 0.0

        overlap = len(t_keywords & c_keywords) / len(t_keywords)
        if overlap >= 0.5:
            return 0.6
        if overlap > 0.0:
            return 0.3
        return 0.0

    def _score_efficiency(self, steps_taken: int, max_steps: int) -> float:
        """
        Smooth efficiency curve: 1.0 at ideal, decays exponentially with oversteps.
        Never falls below 0.05 (avoids zeroing the combined score).
        """
        if steps_taken <= self.ideal_steps:
            return 1.0
        oversteps   = steps_taken - self.ideal_steps
        decay_range = max(1, max_steps - self.ideal_steps)
        score       = math.exp(-2.0 * oversteps / decay_range)
        return max(0.05, round(score, 4))

    def _score_tool_economy(self, tool_cost_sum: float) -> float:
        """
        tool_cost_sum is typically negative (each tool call subtracts a penalty).
        Maps the sum to [0.05, 1.0].
        """
        return max(0.05, min(1.0, 1.0 + tool_cost_sum))

    def _score_tool_diversity(self, tool_names: List[str]) -> float:
        """
        Reward using diverse tools; penalise calling the same tool many times.
        Score = unique_tools / total_calls  (floor 0.05 when total > 0).
        """
        if not tool_names:
            return 1.0
        unique = len(set(tool_names))
        total  = len(tool_names)
        return max(0.05, round(unique / total, 4))

    def _score_cmd_quality(self, tool_names: List[str], target_root_cause: str) -> float:
        """
        Heuristic: reward using tools whose names hint at the root cause.
        E.g. for 'bgp_peer_reset', using 'check_bgp' / 'traceroute' scores higher.
        """
        if not tool_names:
            return 0.5   # neutral when no tool data

        relevant_tools = {
            "dns":      {"nslookup", "check_logs", "check_service"},
            "dhcp":     {"check_dhcp", "arp_scan", "check_logs"},
            "firewall": {"check_iptables", "ping", "traceroute"},
            "ntp":      {"check_ntp", "check_logs", "check_service", "curl"},
            "bgp":      {"check_bgp", "traceroute", "check_logs"},
            "cluster":  {"check_cluster", "check_logs"},
        }

        used     = set(tool_names)
        best_hit = 0.0
        for domain, domain_tools in relevant_tools.items():
            if domain in target_root_cause:
                hits = len(used & domain_tools)
                hit_ratio = hits / max(len(domain_tools), 1)
                best_hit  = max(best_hit, hit_ratio)

        # Default quality if no domain matched
        return max(0.3, round(best_hit, 4)) if best_hit > 0 else 0.5
