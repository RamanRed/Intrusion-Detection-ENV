"""
tool_registry.py — Network diagnostic tool registry (15 tools, 3 new for v3 scenarios).

Tools added in v3:
  - check_routes   (routing_loop scenario)
  - check_replica  (replica_lag scenario)
  - check_queue    (job_queue_stall scenario)
"""

from typing import Callable, Dict, Any, List
import networkx as nx


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        os_support: List[str],
        parameters_schema: Dict[str, Any],
        cost_penalty: float,
        category: str = "general",
    ):
        def decorator(func: Callable):
            self._tools[name] = {
                "name":              name,
                "description":       description,
                "os_support":        os_support,
                "parameters_schema": parameters_schema,
                "cost_penalty":      cost_penalty,
                "category":          category,
                "handler":           func,
            }
            return func
        return decorator

    def get_tool(self, name: str) -> Dict[str, Any]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name":              t["name"],
                "description":       t["description"],
                "category":          t["category"],
                "os_support":        t["os_support"],
                "parameters_schema": t["parameters_schema"],
                "cost_penalty":      t["cost_penalty"],
            }
            for t in self._tools.values()
        ]


tool_registry = ToolRegistry()

ALL_OS     = ["linux", "windows", "macos", "android"]
LINUX_ONLY = ["linux"]


# ═══════════════════════════════════════════════════════════════════════════════
#  CONNECTIVITY
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="ping",
    description="ICMP echo test to a target host. Reveals reachability and latency.",
    os_support=ALL_OS,
    parameters_schema={
        "target": {"type": "string",  "required": True},
        "count":  {"type": "integer", "required": False, "default": 4},
    },
    cost_penalty=-0.05,
    category="connectivity",
)
async def tool_ping(params: Dict, graph: nx.DiGraph) -> Dict:
    target = params.get("target", "")
    count  = int(params.get("count", 4))

    if target not in graph.nodes:
        return {"output": f"ping: {target}: Name or service not known",
                "metadata": {"reachable": False, "reason": "unknown_host"}}

    node = graph.nodes[target]

    if node.get("status") == "down":
        return {
            "output":   (f"PING {target} ({node.get('ip','?')}): 56 data bytes\n"
                         f"Request timeout for icmp_seq 0\n"
                         f"--- {target} ping statistics ---\n"
                         f"{count} packets transmitted, 0 received, 100% packet loss"),
            "metadata": {"reachable": False, "reason": "host_down"},
        }

    if node.get("reachable") is False:
        return {
            "output":   (f"PING {target} ({node.get('ip','?')}): 56 data bytes\n"
                         f"From core-router: Network Unreachable (route withdrawn)\n"
                         f"--- {target} ping statistics ---\n"
                         f"{count} packets transmitted, 0 received, 100% packet loss"),
            "metadata": {"reachable": False, "reason": "no_route"},
        }

    latency = 1.2
    lines   = "\n".join(
        f"64 bytes from {target} ({node.get('ip','?')}): icmp_seq={i} ttl=64 time={latency:.1f} ms"
        for i in range(1, count + 1)
    )
    return {
        "output":   (f"PING {target} ({node.get('ip','?')}): 56 data bytes\n{lines}\n"
                     f"--- {target} ping statistics ---\n"
                     f"{count} packets transmitted, {count} received, 0% packet loss"),
        "metadata": {"reachable": True, "avg_ms": latency},
    }


@tool_registry.register(
    name="traceroute",
    description="Trace the network path hop-by-hop to a destination.",
    os_support=ALL_OS,
    parameters_schema={
        "target":   {"type": "string",  "required": True},
        "max_hops": {"type": "integer", "required": False, "default": 15},
    },
    cost_penalty=-0.10,
    category="connectivity",
)
async def tool_traceroute(params: Dict, graph: nx.DiGraph) -> Dict:
    target = params.get("target", "")

    if target not in graph.nodes:
        return {"output": f"traceroute: {target}: Name or service not known", "metadata": {}}

    node = graph.nodes[target]
    ip   = node.get("ip", "?")

    # Routing loop detection
    if target in ("router-a", "router-b") or graph.nodes.get("router-a", {}).get("routing_loop"):
        return {
            "output": (f"traceroute to {target} ({ip}), 15 hops max\n"
                       f" 1  router-a (10.0.1.1)  1.1 ms\n"
                       f" 2  router-b (10.0.1.2)  1.2 ms\n"
                       f" 3  router-a (10.0.1.1)  1.1 ms  <-- LOOP DETECTED\n"
                       f" 4  router-b (10.0.1.2)  1.2 ms  <-- LOOP DETECTED\n"
                       f" 5  * * *   (TTL expired — routing loop between router-a and router-b)"),
            "metadata": {"routing_loop": True, "loop_nodes": ["router-a", "router-b"]},
        }

    # BGP-caused route loss
    if (target == "db-server"
            and "core-router" in graph.nodes
            and graph.nodes["core-router"].get("bgp_session") == "down"):
        return {
            "output": (f"traceroute to {target} ({ip}), 15 hops max\n"
                       f" 1  app-server (10.1.0.20)  0.5 ms\n"
                       f" 2  core-router (10.1.0.1)  1.2 ms\n"
                       f" 3  * * *   (route withdrawn — no BGP path to 10.1.0.30/24)\n"
                       f" 4  * * *"),
            "metadata": {"route_lost": True, "last_hop": "core-router", "reason": "bgp_route_withdrawn"},
        }

    # Firewall block
    if (target == "internet-router"
            and graph.has_edge("host-a", "internet-router")
            and graph.edges[("host-a", "internet-router")].get("firewall_blocked")):
        return {
            "output": (f"traceroute to {target} ({ip}), 15 hops max\n"
                       f" 1  * * *   (filtered — packet dropped by firewall)\n"
                       f" 2  * * *"),
            "metadata": {"blocked": True, "reason": "firewall_drop"},
        }

    return {
        "output":   f"traceroute to {target} ({ip}), 15 hops max\n 1  {target} ({ip})  1.2 ms",
        "metadata": {"route_ok": True, "hops": 1},
    }


@tool_registry.register(
    name="arp_scan",
    description="Scan the local subnet and list MAC/IP mappings.",
    os_support=LINUX_ONLY,
    parameters_schema={"subnet": {"type": "string", "required": False, "default": "10.0.0.0/24"}},
    cost_penalty=-0.15,
    category="connectivity",
)
async def tool_arp_scan(params: Dict, graph: nx.DiGraph) -> Dict:
    subnet = params.get("subnet", "10.0.0.0/24")
    lines  = [f"Starting arp-scan {subnet}"]
    for node, attrs in graph.nodes(data=True):
        ip = attrs.get("ip", "")
        if ip.startswith("10.0"):
            lines.append(f"{ip}\taa:bb:cc:dd:ee:ff\t{node}")
    dhcp = graph.nodes.get("dhcp-server", {})
    if dhcp.get("pool_exhausted") and dhcp.get("rogue_mac"):
        lines.append(f"10.0.0.99\t{dhcp['rogue_mac']}\t<UNKNOWN> *** HIGH REQUEST RATE ***")
    lines.append(f"\n{len(lines)-1} hosts scanned")
    return {"output": "\n".join(lines), "metadata": {"subnet": subnet}}


# ═══════════════════════════════════════════════════════════════════════════════
#  DNS
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="nslookup",
    description="Query DNS to resolve a hostname or IP address.",
    os_support=ALL_OS,
    parameters_schema={
        "domain": {"type": "string", "required": True},
        "server": {"type": "string", "required": False},
    },
    cost_penalty=-0.05,
    category="dns",
)
async def tool_nslookup(params: Dict, graph: nx.DiGraph) -> Dict:
    domain   = params.get("domain", "")
    dns_node = graph.nodes.get("dns-server", {})
    if dns_node.get("status") == "down":
        return {
            "output":   ";; connection timed out; no servers could be reached\n** server can't find google.com: SERVFAIL",
            "metadata": {"dns_up": False, "reason": "named_down"},
        }
    return {
        "output":   f"Server:  10.0.0.1\nAddress: 10.0.0.1#53\n\nName:\t{domain}\nAddress: 93.184.216.34",
        "metadata": {"dns_up": True, "resolved": True},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  HTTP
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="curl",
    description="Make an HTTP/HTTPS request to a service endpoint.",
    os_support=ALL_OS,
    parameters_schema={
        "url":    {"type": "string", "required": True},
        "method": {"type": "string", "required": False, "default": "GET"},
    },
    cost_penalty=-0.05,
    category="application",
)
async def tool_curl(params: Dict, graph: nx.DiGraph) -> Dict:
    url = params.get("url", "")
    for node_name, attrs in graph.nodes(data=True):
        if attrs.get("tls_errors") and ("https" in url or node_name in url):
            drift = attrs.get("clock_drift_s", 0)
            return {
                "output":   f"curl: (60) SSL certificate problem: certificate is not yet valid (drift={drift}s)",
                "metadata": {"tls_error": True, "clock_drift_s": drift},
            }
    if "web-svc" in graph.nodes and "web-svc" in url:
        if graph.nodes["web-svc"].get("http_status") == 502:
            return {
                "output":   "HTTP/1.1 502 Bad Gateway\r\nX-Upstream-Error: connection refused to app-server:8080",
                "metadata": {"status_code": 502},
            }
    return {"output": "HTTP/1.1 200 OK\r\n\r\n{\"status\":\"ok\"}", "metadata": {"status_code": 200}}


# ═══════════════════════════════════════════════════════════════════════════════
#  SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_service",
    description="Check systemd service status on a host.",
    os_support=LINUX_ONLY,
    parameters_schema={
        "host":    {"type": "string", "required": True},
        "service": {"type": "string", "required": True},
    },
    cost_penalty=-0.10,
    category="service",
)
async def tool_check_service(params: Dict, graph: nx.DiGraph) -> Dict:
    host    = params.get("host", "")
    service = params.get("service", "")
    if host not in graph.nodes:
        return {"output": f"ssh: {host}: No route to host", "metadata": {}}
    node = graph.nodes[host]

    if service in ("named", "bind") and node.get("status") == "down":
        err = node.get("named_error", "config parse error")
        return {"output": f"● named.service: failed\n  Error: {err}", "metadata": {"active": False, "error": err}}

    if service in ("ntpd", "chronyd", "ntp") and node.get("status") == "down":
        err = node.get("ntpd_error", "ntpd: crashed")
        return {"output": f"● ntpd.service: failed\n  Error: {err}", "metadata": {"active": False, "crashed": True}}

    if service in ("postgresql", "db") and node.get("db_pool_exhausted"):
        return {"output": (f"● postgresql.service: active (running)\n"
                           f"  WARNING: pool exhausted ({node.get('pool_used',100)}/{node.get('pool_size',100)} used)"),
                "metadata": {"active": True, "pool_exhausted": True}}

    # Job worker crash-loop
    if "worker" in host and node.get("status") == "crash-loop":
        err      = node.get("crash_error", "unknown crash")
        restarts = node.get("restart_count", 0)
        return {
            "output":   (f"● job-worker.service: activating (auto-restart) — CrashLoopBackOff\n"
                         f"  Restart count : {restarts}\n"
                         f"  Last error    : {err}"),
            "metadata": {"active": False, "crash_loop": True, "restarts": restarts, "error": err},
        }

    return {"output": f"● {service}.service: active (running)\n  No errors detected.", "metadata": {"active": True}}


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGS
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_logs",
    description="Fetch recent log lines from a host via journalctl.",
    os_support=ALL_OS,
    parameters_schema={
        "host":  {"type": "string",  "required": True},
        "lines": {"type": "integer", "required": False, "default": 20},
    },
    cost_penalty=-0.05,
    category="logs",
)
async def tool_check_logs(params: Dict, graph: nx.DiGraph) -> Dict:
    host = params.get("host", "")
    if host not in graph.nodes:
        return {"output": f"journalctl: cannot connect to {host}", "metadata": {}}
    node = graph.nodes[host]

    if host == "dns-server" and node.get("status") == "down":
        err = node.get("named_error", "config parse error")
        return {"output": f"named[999]: {err}\nsystemd[1]: named.service: Failed", "metadata": {"service_failed": "named"}}

    if host == "dhcp-server" and node.get("pool_exhausted"):
        rogue = node.get("rogue_mac", "de:ad:be:ef:00:01")
        return {"output": f"dhcpd: Pool 10.0.0.0/24: 254/254 leases in use\ndhcpd: DHCPDISCOVER from {rogue} rate=1200/min\ndhcpd: WARNING: pool exhaustion",
                "metadata": {"pool_exhausted": True, "rogue_mac": rogue}}

    if host == "ntp-server" and node.get("status") == "down":
        err   = node.get("ntpd_error", "segfault")
        drift = node.get("clock_drift_s", 380)
        return {"output": f"ntpd[777]: {err}\nkernel: Clock drift detected: {drift}s", "metadata": {"ntpd_crashed": True, "clock_drift_s": drift}}

    if host == "core-router" and node.get("bgp_session") == "down":
        peer = node.get("bgp_peer", "10.1.0.254")
        return {"output": f"bgpd: neighbor {peer} Down Hold_Timer_Expired\nbgpd: Removing route 10.1.0.30/24 from RIB",
                "metadata": {"bgp_down": True, "peer": peer}}

    if host == "app-server" and node.get("db_pool_exhausted"):
        return {"output": "app: connection pool timeout after 30s\napp: upstream db unreachable — returning 502",
                "metadata": {"db_unreachable": True}}

    if host.startswith("cluster-node") and node.get("split_brain"):
        return {"output": f"raft: WARN: granted vote to self — starting election\nraft: SPLIT-BRAIN detected — two leaders in term 7",
                "metadata": {"split_brain": True}}

    # Routing loop
    if host in ("router-a", "router-b") and node.get("routing_loop"):
        peer = "router-b" if host == "router-a" else "router-a"
        return {"output": (f"kernel: ttl exceeded in transit (10.2.0.0/24 → {peer} → {host} → {peer})\n"
                           f"kernel: routing loop detected for 10.2.0.0/24\n"
                           f"ip route: 10.2.0.0/24 via {peer} — next-hop also points back"),
                "metadata": {"routing_loop": True, "affected_prefix": "10.2.0.0/24"}}

    # Replica lag
    if host == "db-replica" and node.get("replica_lag_s", 0) > 0:
        err = node.get("io_error", "binlog position mismatch")
        lag = node.get("replica_lag_s", 0)
        return {"output": (f"mysqld: Slave I/O thread: error {err}\n"
                           f"mysqld: Seconds_Behind_Master: {lag}\n"
                           f"mysqld: IO thread stopped — manual intervention required"),
                "metadata": {"io_error": err, "lag_s": lag}}

    # Job worker crash
    if host == "job-worker" and node.get("status") == "crash-loop":
        err      = node.get("crash_error", "unknown crash")
        restarts = node.get("restart_count", 0)
        return {"output": (f"job-worker: {err}\n"
                           f"systemd: job-worker.service: Failed — restarting ({restarts} times)\n"
                           f"job-worker: No consumers active — queue depth growing"),
                "metadata": {"crash_loop": True, "restarts": restarts, "error": err}}

    return {"output": f"Apr 08 13:44:00 {host} kernel: System operating normally. No errors.", "metadata": {"normal": True}}


# ═══════════════════════════════════════════════════════════════════════════════
#  FIREWALL
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_iptables",
    description="List iptables rules on a host.",
    os_support=LINUX_ONLY,
    parameters_schema={
        "host":  {"type": "string", "required": True},
        "chain": {"type": "string", "required": False, "default": "all"},
    },
    cost_penalty=-0.15,
    category="firewall",
)
async def tool_check_iptables(params: Dict, graph: nx.DiGraph) -> Dict:
    host = params.get("host", "")
    if host not in graph.nodes:
        return {"output": f"ssh: {host}: No route to host", "metadata": {}}
    for src, dst, attrs in graph.edges(data=True):
        if dst == host and attrs.get("firewall_blocked"):
            rule = attrs.get("iptables_rule", "-A FORWARD -j DROP")
            return {
                "output":   (f"Chain FORWARD (policy ACCEPT)\n"
                             f"DROP    all  --  10.0.0.5  0.0.0.0/0  *** SUSPICIOUS RULE ***\n"
                             f"  ({rule})"),
                "metadata": {"firewall_blocked": True, "rule": rule},
            }
    return {"output": "Chain FORWARD (policy ACCEPT)\n  No DROP rules detected.", "metadata": {"firewall_blocked": False}}


# ═══════════════════════════════════════════════════════════════════════════════
#  DHCP
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_dhcp",
    description="Inspect DHCP lease table and pool utilisation.",
    os_support=LINUX_ONLY,
    parameters_schema={"host": {"type": "string", "required": True}},
    cost_penalty=-0.10,
    category="dhcp",
)
async def tool_check_dhcp(params: Dict, graph: nx.DiGraph) -> Dict:
    host = params.get("host", "")
    if host not in graph.nodes:
        return {"output": f"ssh: {host}: No route to host", "metadata": {}}
    node = graph.nodes[host]
    if node.get("pool_exhausted"):
        total = node.get("leases_total", 254)
        used  = node.get("leases_used", 254)
        rogue = node.get("rogue_mac", "de:ad:be:ef:00:01")
        rate  = node.get("rogue_req_rate", 1200)
        return {
            "output":   (f"DHCP Pool 10.0.0.0/24: {used}/{total} leases used\nFree: 0\n"
                         f"Rogue MAC: {rogue}  {rate} req/min  <-- ANOMALOUS\nStatus: POOL EXHAUSTED"),
            "metadata": {"pool_exhausted": True, "used": used, "total": total, "rogue_mac": rogue},
        }
    return {"output": "DHCP Pool 10.0.0.0/24: 42/254 leases used\nStatus: OK", "metadata": {"pool_exhausted": False}}


# ═══════════════════════════════════════════════════════════════════════════════
#  NTP
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_ntp",
    description="Check NTP sync status and clock drift on a host.",
    os_support=LINUX_ONLY,
    parameters_schema={"host": {"type": "string", "required": True}},
    cost_penalty=-0.05,
    category="time",
)
async def tool_check_ntp(params: Dict, graph: nx.DiGraph) -> Dict:
    host = params.get("host", "")
    if host not in graph.nodes:
        return {"output": f"ssh: {host}: No route to host", "metadata": {}}
    node = graph.nodes[host]
    if host == "ntp-server" and node.get("status") == "down":
        drift = node.get("clock_drift_s", 380)
        return {"output": f"chronyc: Cannot talk to daemon\n  ntpd.service: inactive (dead)\n  Clock drift: +{drift}s",
                "metadata": {"ntp_running": False, "drift_s": drift}}
    if node.get("tls_errors") and node.get("clock_drift_s"):
        drift = node["clock_drift_s"]
        return {"output": f"chronyc tracking:\n  Stratum: 16 (unsynchronised)\n  System time: +{drift}s fast",
                "metadata": {"ntp_synced": False, "drift_s": drift}}
    return {"output": "chronyc tracking:\n  Stratum: 2\n  System time: +0.000123s", "metadata": {"ntp_synced": True}}


# ═══════════════════════════════════════════════════════════════════════════════
#  BGP
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_bgp",
    description="Show BGP peer status and routing table on a router.",
    os_support=LINUX_ONLY,
    parameters_schema={"host": {"type": "string", "required": True}},
    cost_penalty=-0.10,
    category="routing",
)
async def tool_check_bgp(params: Dict, graph: nx.DiGraph) -> Dict:
    host = params.get("host", "")
    if host not in graph.nodes:
        return {"output": f"ssh: {host}: No route to host", "metadata": {}}
    node = graph.nodes[host]
    if node.get("bgp_session") == "down":
        peer   = node.get("bgp_peer", "10.1.0.254")
        reason = node.get("bgp_reset_reason", "hold_timer_expired")
        return {
            "output":   (f"Neighbor {peer}: State={reason.upper()}\n"
                         f"WARNING: Route 10.1.0.30/24 removed from RIB (no valid path)"),
            "metadata": {"bgp_up": False, "peer": peer, "state": reason, "routes_withdrawn": ["10.1.0.30/24"]},
        }
    return {"output": f"Neighbor 10.1.0.254: State=ESTABLISHED\nTotal prefixes: 12", "metadata": {"bgp_up": True}}


# ═══════════════════════════════════════════════════════════════════════════════
#  CLUSTER
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_cluster",
    description="Inspect distributed cluster membership, leader status, and quorum.",
    os_support=LINUX_ONLY,
    parameters_schema={"host": {"type": "string", "required": True}},
    cost_penalty=-0.10,
    category="cluster",
)
async def tool_check_cluster(params: Dict, graph: nx.DiGraph) -> Dict:
    host = params.get("host", "")
    if host not in graph.nodes:
        return {"output": f"ssh: {host}: No route to host", "metadata": {}}
    cluster_nodes = {n: d for n, d in graph.nodes(data=True) if n.startswith("cluster-node")}
    if not cluster_nodes:
        return {"output": "No cluster nodes found in topology.", "metadata": {}}
    leaders = [n for n, d in cluster_nodes.items() if d.get("cluster_role") == "leader"]
    split   = len(leaders) > 1
    lines   = [f"Cluster membership view from {host}:", ""]
    for name, attrs in sorted(cluster_nodes.items()):
        flag = "  <-- STALE LEADER (SPLIT-BRAIN)" if attrs.get("split_brain") else ""
        lines.append(f"  {name}  role={attrs.get('cluster_role','?')}  heartbeat_timeout={attrs.get('heartbeat_timeout_ms','?')}ms{flag}")
    lines.append("")
    if split:
        lines.append(f"ALERT: SPLIT-BRAIN — {len(leaders)} leaders: {leaders}")
    else:
        lines.append(f"Quorum: OK  |  Leader: {leaders[0] if leaders else 'none'}")
    return {"output": "\n".join(lines), "metadata": {"split_brain": split, "leaders": leaders}}


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTING TABLE (v3 — for routing_loop scenario)
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_routes",
    description="Show IP routing table on a router ('ip route show'). Detects loops and bad static routes.",
    os_support=LINUX_ONLY,
    parameters_schema={"host": {"type": "string", "required": True}},
    cost_penalty=-0.10,
    category="routing",
)
async def tool_check_routes(params: Dict, graph: nx.DiGraph) -> Dict:
    host = params.get("host", "")
    if host not in graph.nodes:
        return {"output": f"ssh: {host}: No route to host", "metadata": {}}
    node = graph.nodes[host]
    if node.get("routing_loop"):
        bad_routes = [
            attrs["bad_static_route"]
            for src, dst, attrs in graph.edges(data=True)
            if src == host and attrs.get("bad_static_route")
        ]
        peer    = "router-b" if host == "router-a" else "router-a"
        peer_ip = graph.nodes.get(peer, {}).get("ip", "?")
        bad_str = "\n".join(f"  {r}  <-- LOOP (next-hop also routes back)" for r in bad_routes)
        return {
            "output":   (f"# ip route show on {host} ({node.get('ip','?')})\n"
                         f"default via 10.0.0.1 dev eth0\n"
                         f"10.0.0.0/24 dev eth0 proto kernel\n"
                         f"10.2.0.0/24 via {peer_ip} dev eth1  <-- STATIC ROUTE (suspect)\n\n"
                         f"WARNING: Routing loop detected:\n{bad_str}"),
            "metadata": {"routing_loop": True, "bad_routes": bad_routes},
        }
    return {
        "output":   (f"# ip route show on {host} ({node.get('ip','?')})\n"
                     f"default via 10.0.0.1 dev eth0\n"
                     f"10.0.0.0/24 dev eth0 proto kernel\n"
                     f"No suspicious static routes."),
        "metadata": {"routing_loop": False},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE REPLICATION (v3 — for replica_lag scenario)
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_replica",
    description="Run SHOW SLAVE STATUS on a MySQL read replica to check replication health.",
    os_support=LINUX_ONLY,
    parameters_schema={"host": {"type": "string", "required": True}},
    cost_penalty=-0.10,
    category="database",
)
async def tool_check_replica(params: Dict, graph: nx.DiGraph) -> Dict:
    host = params.get("host", "")
    if host not in graph.nodes:
        return {"output": f"ssh: {host}: No route to host", "metadata": {}}
    node    = graph.nodes[host]
    lag     = node.get("replica_lag_s", 0)
    io_err  = node.get("io_error", "")
    binlog  = node.get("binlog_file", "mysql-bin.000001")
    binpos  = node.get("binlog_pos", 0)
    if io_err:
        return {
            "output":   (f"Slave_IO_Running: Error  <-- IO THREAD FAILED\n"
                         f"Slave_SQL_Running: Yes\n"
                         f"Seconds_Behind_Master: {lag}\n"
                         f"Master_Log_File: {binlog}\n"
                         f"Read_Master_Log_Pos: {binpos}\n"
                         f"Last_IO_Error: {io_err}\n\n"
                         f"ACTION: IO thread stopped — binlog position mismatch."),
            "metadata": {"replica_healthy": False, "io_thread_error": io_err, "lag_seconds": lag},
        }
    return {
        "output":   "Slave_IO_Running: Yes\nSlave_SQL_Running: Yes\nSeconds_Behind_Master: 0\nReplication OK.",
        "metadata": {"replica_healthy": True, "lag_seconds": 0},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  JOB QUEUE (v3 — for job_queue_stall scenario)
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_queue",
    description="Inspect Redis job queue depth and active worker consumer count.",
    os_support=LINUX_ONLY,
    parameters_schema={
        "host":  {"type": "string", "required": True},
        "queue": {"type": "string", "required": False, "default": "jobs:default"},
    },
    cost_penalty=-0.05,
    category="queue",
)
async def tool_check_queue(params: Dict, graph: nx.DiGraph) -> Dict:
    host  = params.get("host", "")
    queue = params.get("queue", "jobs:default")
    if host not in graph.nodes:
        return {"output": f"redis-cli: {host}: Connection refused", "metadata": {}}
    node    = graph.nodes[host]
    depth   = node.get("queue_depth", 0)
    workers = {n: d for n, d in graph.nodes(data=True) if "worker" in n}
    crashes = [n for n, d in workers.items() if d.get("status") == "crash-loop"]
    if depth > 1000 and crashes:
        crash_err = workers[crashes[0]].get("crash_error", "unknown") if crashes else ""
        restarts  = sum(d.get("restart_count", 0) for d in workers.values())
        return {
            "output":   (f"redis-cli LLEN {queue}: {depth}  <-- HIGH DEPTH\n"
                         f"Active consumers: 0  (workers not consuming)\n\n"
                         f"Crash workers: {crashes}\n"
                         f"  Total restarts: {restarts}\n"
                         f"  Last error: {crash_err}\n\n"
                         f"WARNING: {depth} pending jobs, no active consumers."),
            "metadata": {"queue_depth": depth, "consumers": 0, "crash_workers": crashes, "crash_error": crash_err},
        }
    return {
        "output":   f"redis-cli LLEN {queue}: {depth}\nConsumers: {max(1, len(workers))}\nQueue OK",
        "metadata": {"queue_depth": depth, "consumers": len(workers)},
    }
