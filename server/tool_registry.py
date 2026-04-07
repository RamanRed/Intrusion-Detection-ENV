from typing import Callable, Dict, Any, List
import networkx as nx


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, os_support: List[str],
                 parameters_schema: Dict[str, Any], cost_penalty: float):
        def decorator(func: Callable):
            self._tools[name] = {
                "name": name,
                "description": description,
                "os_support": os_support,
                "parameters_schema": parameters_schema,
                "cost_penalty": cost_penalty,
                "handler": func,
            }
            return func
        return decorator

    def get_tool(self, name: str) -> Dict[str, Any]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": t["name"], "description": t["description"],
             "parameters_schema": t["parameters_schema"]}
            for t in self._tools.values()
        ]


tool_registry = ToolRegistry()

ALL_OS = ["linux", "windows", "macos", "android"]


# ── ping ─────────────────────────────────────────────────────────────────────

@tool_registry.register(
    name="ping",
    description="ICMP connectivity test between two hosts",
    os_support=ALL_OS,
    parameters_schema={"target": {"type": "string", "required": True},
                       "count":  {"type": "integer", "default": 4}},
    cost_penalty=-0.1,
)
async def tool_ping(params: Dict, graph: nx.DiGraph) -> Dict:
    target = params.get("target", "")
    if target not in graph.nodes:
        return {"output": f"ping: {target}: Name or service not known"}
    if graph.nodes[target].get("status") == "down":
        return {"output": f"ping: {target}: Destination Host Unreachable"}
    if graph.nodes[target].get("reachable") is False:
        return {"output": f"ping: {target}: Network Unreachable (routing issue detected)"}
    return {"output": (f"PING {target}: 56 data bytes\n"
                       f"64 bytes from {target}: icmp_seq=1 ttl=64 time=1.2 ms\n"
                       f"64 bytes from {target}: icmp_seq=2 ttl=64 time=1.1 ms")}


# ── nslookup ─────────────────────────────────────────────────────────────────

@tool_registry.register(
    name="nslookup",
    description="DNS name resolution query",
    os_support=ALL_OS,
    parameters_schema={"domain": {"type": "string", "required": True}},
    cost_penalty=-0.1,
)
async def tool_nslookup(params: Dict, graph: nx.DiGraph) -> Dict:
    domain = params.get("domain", "")
    if "dns-server" in graph.nodes and graph.nodes["dns-server"].get("status") == "down":
        return {"output": ";; connection timed out; no servers could be reached"}
    return {"output": (f"Server:  10.0.0.1\nAddress: 10.0.0.1#53\n\n"
                       f"Non-authoritative answer:\nName: {domain}\nAddress: 93.184.216.34")}


# ── curl ──────────────────────────────────────────────────────────────────────

@tool_registry.register(
    name="curl",
    description="Make an HTTP request to a service endpoint",
    os_support=ALL_OS,
    parameters_schema={"url": {"type": "string", "required": True}},
    cost_penalty=-0.1,
)
async def tool_curl(params: Dict, graph: nx.DiGraph) -> Dict:
    url = params.get("url", "")
    # Detect web-svc 502
    if "web-svc" in graph.nodes and "web-svc" in url:
        if graph.nodes["web-svc"].get("http_status") == 502:
            return {"output": "HTTP/1.1 502 Bad Gateway\nContent-Type: text/html\n\n<html>502 Bad Gateway</html>"}
    return {"output": f"HTTP/1.1 200 OK\nContent-Length: 42\n\nOK"}


# ── check_service ─────────────────────────────────────────────────────────────

@tool_registry.register(
    name="check_service",
    description="Check the status of a named service on a host (e.g. systemctl status)",
    os_support=["linux"],
    parameters_schema={"host":    {"type": "string", "required": True},
                       "service": {"type": "string", "required": True}},
    cost_penalty=-0.15,
)
async def tool_check_service(params: Dict, graph: nx.DiGraph) -> Dict:
    host    = params.get("host", "")
    service = params.get("service", "")
    if host not in graph.nodes:
        return {"output": f"Error: host '{host}' not found in topology"}
    node = graph.nodes[host]
    if service == "named" and node.get("status") == "down":
        return {"output": f"● named.service - BIND Domain Name Server\n   Loaded: loaded\n   Active: failed (Result: exit-code)"}
    if service == "db" and node.get("db_pool_exhausted"):
        return {"output": f"● postgresql.service\n   Active: active (running)\n   Warning: connection pool exhausted (0/100 connections available)"}
    if service == "bgp" and node.get("bgp_session") == "down":
        return {"output": f"BGP session to peer 10.1.0.254: IDLE (last reset: peer hold-timer expired)"}
    return {"output": f"● {service}.service\n   Active: active (running) since 2026-04-07"}


# ── traceroute ────────────────────────────────────────────────────────────────

@tool_registry.register(
    name="traceroute",
    description="Trace the network path to a destination host",
    os_support=ALL_OS,
    parameters_schema={"target": {"type": "string", "required": True}},
    cost_penalty=-0.15,
)
async def tool_traceroute(params: Dict, graph: nx.DiGraph) -> Dict:
    target = params.get("target", "")
    if target not in graph.nodes:
        return {"output": f"traceroute: {target}: Name or service not known"}
    # Cascading failure: route to db-server is lost
    if target == "db-server" and "core-router" in graph.nodes:
        if graph.nodes["core-router"].get("bgp_session") == "down":
            return {"output": ("traceroute to db-server (10.1.0.30)\n"
                               " 1  app-server (10.1.0.20)  0.5 ms\n"
                               " 2  core-router (10.1.0.1)  1.2 ms\n"
                               " 3  * * *  (route lost — no BGP path to 10.1.0.30/24)")}
    return {"output": f"traceroute to {target}\n 1  {target}  1.2 ms  1.1 ms  1.0 ms"}


# ── check_logs ────────────────────────────────────────────────────────────────

@tool_registry.register(
    name="check_logs",
    description="Read recent log lines from a host (journalctl / event viewer)",
    os_support=ALL_OS,
    parameters_schema={"host":  {"type": "string", "required": True},
                       "lines": {"type": "integer", "default": 20}},
    cost_penalty=-0.1,
)
async def tool_check_logs(params: Dict, graph: nx.DiGraph) -> Dict:
    host = params.get("host", "")
    if host not in graph.nodes:
        return {"output": f"Error: host '{host}' not in topology"}
    node = graph.nodes[host]
    if host == "core-router" and node.get("bgp_session") == "down":
        return {"output": ("Apr 07 13:42:01 core-router bgpd[1234]: %BGP-5-ADJCHANGE: "
                           "neighbor 10.1.0.254 Down Hold Timer Expired\n"
                           "Apr 07 13:42:01 core-router bgpd[1234]: Removing route 10.1.0.30/24 from RIB")}
    if host == "app-server" and node.get("db_pool_exhausted"):
        return {"output": ("Apr 07 13:43:15 app-server app[5678]: ERROR: connection pool timeout "
                           "after 30s — no available connections to db-server:5432\n"
                           "Apr 07 13:43:15 app-server app[5678]: FATAL: upstream db unreachable")}
    if host == "dns-server" and node.get("status") == "down":
        return {"output": ("Apr 07 13:40:00 dns-server named[999]: "
                           "loading configuration from '/etc/named.conf'\n"
                           "Apr 07 13:40:00 dns-server named[999]: /etc/named.conf:14: "
                           "unknown option 'recursion-limit'\n"
                           "Apr 07 13:40:00 dns-server named[999]: exiting (due to fatal error)")}
    return {"output": f"Apr 07 13:44:00 {host} kernel: Normal operation, no errors."}
