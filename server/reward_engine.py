from typing import Dict, Any

class RewardEngine:
    def __init__(self, difficulty: str):
        self.difficulty_multiplier = {
            "easy": 1.0,
            "medium": 1.2,
            "hard": 1.5,
            "expert": 2.0
        }.get(difficulty, 1.0)
        
    def compute(self, target_root_cause: str, claimed_cause: str, is_resolved: bool, tools_called: int, tool_cost_sum: float) -> float:
        r_resolution = 1.0 if is_resolved else 0.0
        
        # Simple string match or basic proximity for RCA grading
        r_rca = 1.0 if claimed_cause.lower() in target_root_cause.lower() or target_root_cause.lower() in claimed_cause.lower() else 0.0
        
        # Efficiency assumes ideal path is short. Say max 50 steps:
        steps_taken = tools_called
        r_efficiency = max(0.0, 1.0 - (steps_taken / 50.0))
        
        # tool cost based strictly on penalty accumulated
        r_tool_cost = max(0.0, 1.0 + tool_cost_sum) # Sum is typically negative
        
        # Assuming no safety violations here to simplify
        r_safety = 1.0 
        r_cmd_quality = 0.5 # Default middle LLM score without actual LLM
        
        r_total = (
            0.40 * r_resolution +
            0.25 * r_rca +
            0.15 * r_efficiency +
            0.10 * r_tool_cost +
            0.05 * r_safety +
            0.05 * r_cmd_quality
        ) * self.difficulty_multiplier
        
        return r_total

    def compute_penalty_violation(self):
        return -10.0
