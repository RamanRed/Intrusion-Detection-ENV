from typing import Callable, Dict, Any, List
import networkx as nx

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, os_support: List[str], parameters_schema: Dict[str, Any], cost_penalty: float):
        def decorator(func: Callable):
            self._tools[name] = {
                "name": name,
                "description": description,
                "os_support": os_support,
                "parameters_schema": parameters_schema,
                "cost_penalty": cost_penalty,
                "handler": func
            }
            return func
        return decorator

    def get_tool(self, name: str) -> Dict[str, Any]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [{"name": t["name"], "description": t["description"], "parameters_schema": t["parameters_schema"]} for t in self._tools.values()]

tool_registry = ToolRegistry()

# -----------------
# Built-in Tools
# -----------------

@tool_registry.register(
    name="ping",
    description="ICMP connectivity test",
    os_support=["linux", "windows", "macos", "android"],
    parameters_schema={"target": {"type": "string", "required": "true"}, "count": {"type": "integer", "default": 4}},
    cost_penalty=-0.1
)
async def tool_ping(params: Dict, graph: nx.DiGraph) -> Dict:
    target = params.get("target")
    if target in graph.nodes:
        if graph.nodes[target].get("status") == "down":
            return {"output": f"ping: {target} is unreachable."}
        return {"output": f"PING {target} 56(84) bytes of data.\n64 bytes from {target}: icmp_seq=1 ttl=64 time=1.2 ms"}
    return {"output": f"ping: {target} Name or service not known"}

@tool_registry.register(
    name="nslookup",
    description="DNS name resolution",
    os_support=["linux", "windows", "macos", "android"],
    parameters_schema={"domain": {"type": "string", "required": "true"}},
    cost_penalty=-0.1
)
async def tool_nslookup(params: Dict, graph: nx.DiGraph) -> Dict:
    domain = params.get("domain")
    # Simulate a DNS lookup using graph properties
    if "dns-server" in graph.nodes and graph.nodes["dns-server"].get("status") == "down":
        return {"output": ";; connection timed out; no servers could be reached"}
    return {"output": f"Server: 10.0.0.1\nAddress: 10.0.0.1#53\n\nName: {domain}\nAddress: 93.184.216.34"}
