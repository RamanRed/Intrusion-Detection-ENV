import networkx as nx
from typing import Dict, Any, Tuple

class ScenarioGenerator:
    def __init__(self):
        pass

    def generate(self, scenario_id: str, os_profile: str, difficulty: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        graph = nx.DiGraph()
        
        # Base setup
        graph.add_node("host-a", os=os_profile, ip="10.0.0.5", status="up")
        graph.add_node("dns-server", os="linux", ip="10.0.0.1", status="up")
        graph.add_node("internet-router", os="linux", ip="10.0.0.254", status="up")
        
        graph.add_edge("host-a", "dns-server", latency_ms=1)
        graph.add_edge("host-a", "internet-router", latency_ms=2)

        ground_truth = {
            "scenario_id": scenario_id,
            "root_cause_node": "",
            "root_cause": "",
            "fix_applied": ""
        }

        # Apply specific scenario faults
        if scenario_id == "dns_failure":
            graph.nodes["dns-server"]["status"] = "down"
            ground_truth["root_cause_node"] = "dns-server"
            ground_truth["root_cause"] = "dns_misconfiguration"
            ground_truth["fix_applied"] = "restarted_named"
        elif scenario_id == "firewall_block":
            graph.edges[("host-a", "internet-router")]["firewall_blocked"] = True
            ground_truth["root_cause_node"] = "internet-router"
            ground_truth["root_cause"] = "firewall_rule_drop"
            ground_truth["fix_applied"] = "iptables_flush"
        else:
            # Default to a random baseline scenario
            pass
            
        return graph, ground_truth
