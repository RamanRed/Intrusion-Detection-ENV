"""
tool_registry.py — Advanced network diagnostic tool registry.

Improvements over v1:
  - 12 tools (was 6): added arp_scan, check_iptables, check_dhcp, check_ntp,
    check_bgp, check_cluster
  - Every tool returns structured `metadata` alongside `output` string
  - Richer realistic output with IP addresses, timestamps, error codes
  - Partial-observability: graph attributes drive what each tool reveals
  - Cost penalties scaled realistically by tool intrusiveness
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
        "target": {"type": "string",  "required": True,  "description": "Hostname or IP"},
        "count":  {"type": "integer", "required": False, "default": 4},
    },
    cost_penalty=-0.05,
    category="connectivity",
)
async def tool_ping(params: Dict, graph: nx.DiGraph) -> Dict:
    target = params.get("target", "")
    count  = int(params.get("count", 4))

    if target not in graph.nodes:
        return {
            "output":   f"ping: {target}: Name or service not known",
            "metadata": {"reachable": False, "reason": "unknown_host"},
        }

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
        "output": (f"PING {target} ({node.get('ip','?')}): 56 data bytes\n{lines}\n"
                   f"--- {target} ping statistics ---\n"
                   f"{count} packets transmitted, {count} received, 0% packet loss"),
        "metadata": {"reachable": True, "avg_ms": latency},
    }


@tool_registry.register(
    name="traceroute",
    description="Trace the network path hop-by-hop to a destination.",
    os_support=ALL_OS,
    parameters_schema={
        "target":    {"type": "string",  "required": True},
        "max_hops":  {"type": "integer", "required": False, "default": 15},
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

    # Cascading failure: route to db-server is lost via core-router BGP down
    if (target == "db-server"
            and "core-router" in graph.nodes
            and graph.nodes["core-router"].get("bgp_session") == "down"):
        return {
            "output": (f"traceroute to {target} ({ip}), 15 hops max, 60 byte packets\n"
                       f" 1  app-server (10.1.0.20)  0.5 ms  0.4 ms  0.5 ms\n"
                       f" 2  core-router (10.1.0.1)  1.2 ms  1.1 ms  1.3 ms\n"
                       f" 3  * * *   (route withdrawn — no BGP path to 10.1.0.30/24)\n"
                       f" 4  * * *\n"
                       f" 5  * * *"),
            "metadata": {"route_lost": True, "last_hop": "core-router", "reason": "bgp_route_withdrawn"},
        }

    # Firewall blocks internet-router path
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
        "output": (f"traceroute to {target} ({ip}), 15 hops max\n"
                   f" 1  {target} ({ip})  1.2 ms  1.1 ms  1.0 ms"),
        "metadata": {"route_ok": True, "hops": 1},
    }


@tool_registry.register(
    name="arp_scan",
    description="Scan the local subnet and list MAC/IP mappings (requires root).",
    os_support=LINUX_ONLY,
    parameters_schema={
        "subnet": {"type": "string", "required": False, "default": "10.0.0.0/24"},
    },
    cost_penalty=-0.15,
    category="connectivity",
)
async def tool_arp_scan(params: Dict, graph: nx.DiGraph) -> Dict:
    subnet = params.get("subnet", "10.0.0.0/24")
    lines  = ["Interface: eth0, datalink type: EN10MB (Ethernet)",
              f"Starting arp-scan {subnet}"]
    for node, attrs in graph.nodes(data=True):
        ip = attrs.get("ip", "")
        if ip.startswith("10.0"):
            mac = attrs.get("mac", "aa:bb:cc:dd:ee:ff")
            lines.append(f"{ip}\t{mac}\t{node}")

    # Highlight rogue device if dhcp starvation scenario
    dhcp = graph.nodes.get("dhcp-server", {})
    if dhcp.get("pool_exhausted") and dhcp.get("rogue_mac"):
        lines.append(f"10.0.0.99\t{dhcp['rogue_mac']}\t<UNKNOWN> *** HIGH REQUEST RATE ***")

    lines.append(f"\n{len(lines)-2} hosts scanned")
    return {"output": "\n".join(lines), "metadata": {"subnet": subnet}}


# ═══════════════════════════════════════════════════════════════════════════════
#  DNS / NAME RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="nslookup",
    description="Query DNS to resolve a hostname or IP address.",
    os_support=ALL_OS,
    parameters_schema={
        "domain":  {"type": "string", "required": True},
        "server":  {"type": "string", "required": False, "description": "DNS server IP"},
    },
    cost_penalty=-0.05,
    category="dns",
)
async def tool_nslookup(params: Dict, graph: nx.DiGraph) -> Dict:
    domain = params.get("domain", "")

    dns_node = graph.nodes.get("dns-server", {})
    if dns_node.get("status") == "down":
        return {
            "output":   (";; connection timed out; no servers could be reached\n"
                         "** server can't find google.com: SERVFAIL"),
            "metadata": {"dns_up": False, "reason": "named_down"},
        }

    return {
        "output": (f"Server:  10.0.0.1\nAddress: 10.0.0.1#53\n\n"
                   f"Non-authoritative answer:\nName:\t{domain}\nAddress: 93.184.216.34"),
        "metadata": {"dns_up": True, "resolved": True},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  HTTP / APPLICATION
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

    # TLS errors from NTP drift
    for node_name, attrs in graph.nodes(data=True):
        if attrs.get("tls_errors") and ("https" in url or node_name in url):
            drift = attrs.get("clock_drift_s", 0)
            return {
                "output":   (f"curl: (60) SSL certificate problem: certificate is not yet valid\n"
                             f"  Certificate's notBefore: 2026-04-08T12:00:00Z\n"
                             f"  Current time (local):    2026-04-08T05:37:00Z  (drift={drift}s)\n"
                             f"curl: (60) Closing connection 0"),
                "metadata": {"tls_error": True, "clock_drift_s": drift},
            }

    # 502 from web-svc
    if "web-svc" in graph.nodes and "web-svc" in url:
        if graph.nodes["web-svc"].get("http_status") == 502:
            return {
                "output": ("HTTP/1.1 502 Bad Gateway\r\n"
                           "Content-Type: text/html\r\n"
                           "X-Upstream-Error: connection refused to app-server:8080\r\n\r\n"
                           "<html><body><h1>502 Bad Gateway</h1>"
                           "The upstream server returned an invalid response.</body></html>"),
                "metadata": {"status_code": 502, "upstream_error": "connection_refused"},
            }

    return {
        "output": f"HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\n{{\"status\":\"ok\"}}",
        "metadata": {"status_code": 200},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SERVICE / PROCESS
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_service",
    description="Check systemd/service status on a host (mirrors `systemctl status`).",
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
        err = node.get("named_error", "unknown error")
        return {
            "output": (f"● named.service - BIND Domain Name Server\n"
                       f"   Loaded: loaded (/lib/systemd/system/named.service)\n"
                       f"   Active: failed (Result: exit-code) since 2026-04-08 13:40:00\n"
                       f"  Process: 999 ExecStart=/usr/sbin/named (code=exited, status=1/FAILURE)\n"
                       f" Apr 08 13:40:00 dns-server named[999]: {err}\n"
                       f" Apr 08 13:40:00 dns-server systemd[1]: named.service: Main process exited, code=exited"),
            "metadata": {"active": False, "exit_code": 1, "error": err},
        }

    if service in ("ntpd", "chronyd", "ntp") and node.get("status") == "down":
        err = node.get("ntpd_error", "ntpd: crashed")
        return {
            "output": (f"● ntpd.service - Network Time Protocol Daemon\n"
                       f"   Active: failed (Result: core-dump)\n"
                       f" Apr 08 13:38:00 ntp-server ntpd[777]: {err}"),
            "metadata": {"active": False, "crashed": True, "error": err},
        }

    if service in ("postgresql", "db") and node.get("db_pool_exhausted"):
        return {
            "output": (f"● postgresql.service\n"
                       f"   Active: active (running) since 2026-04-08 12:00:00\n"
                       f"   Warning: connection pool exhausted "
                       f"({node.get('pool_used',100)}/{node.get('pool_size',100)} connections used)\n"
                       f" Apr 08 13:43:00 app-server app[5678]: FATAL connection pool timeout"),
            "metadata": {"active": True, "pool_exhausted": True,
                         "pool_used": node.get("pool_used", 100),
                         "pool_size": node.get("pool_size", 100)},
        }

    return {
        "output": (f"● {service}.service\n"
                   f"   Active: active (running) since 2026-04-08 08:00:00\n"
                   f"   No errors detected."),
        "metadata": {"active": True},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGS
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_logs",
    description="Fetch recent log lines from a host via journalctl / syslog.",
    os_support=ALL_OS,
    parameters_schema={
        "host":  {"type": "string",  "required": True},
        "lines": {"type": "integer", "required": False, "default": 20},
        "unit":  {"type": "string",  "required": False, "description": "systemd unit filter (optional)"},
    },
    cost_penalty=-0.05,
    category="logs",
)
async def tool_check_logs(params: Dict, graph: nx.DiGraph) -> Dict:
    host = params.get("host", "")
    unit = params.get("unit", "")

    if host not in graph.nodes:
        return {"output": f"journalctl: cannot connect to {host}: host not in topology", "metadata": {}}

    node = graph.nodes[host]

    # DNS server named crash
    if host == "dns-server" and node.get("status") == "down":
        err = node.get("named_error", "config parse error")
        return {
            "output": (f"-- Logs begin at Mon 2026-04-08 08:00:00 UTC --\n"
                       f"Apr 08 13:40:00 dns-server named[999]: loading /etc/named.conf\n"
                       f"Apr 08 13:40:00 dns-server named[999]: {err}\n"
                       f"Apr 08 13:40:00 dns-server named[999]: exiting (fatal error in configuration)\n"
                       f"Apr 08 13:40:01 dns-server systemd[1]: named.service: Failed with result 'exit-code'"),
            "metadata": {"service_failed": "named", "error": err},
        }

    # DHCP pool exhaustion
    if host == "dhcp-server" and node.get("pool_exhausted"):
        rogue = node.get("rogue_mac", "de:ad:be:ef:00:01")
        rate  = node.get("rogue_req_rate", 1200)
        return {
            "output": (f"Apr 08 13:41:00 dhcp-server dhcpd[1100]: Pool 10.0.0.0/24: 254/254 leases in use\n"
                       f"Apr 08 13:41:01 dhcp-server dhcpd[1100]: DHCPDISCOVER from {rogue} rate={rate}/min\n"
                       f"Apr 08 13:41:01 dhcp-server dhcpd[1100]: no free leases — DHCPNAK sent\n"
                       f"Apr 08 13:41:02 dhcp-server dhcpd[1100]: WARNING: pool exhaustion — rogue MAC suspected"),
            "metadata": {"pool_exhausted": True, "rogue_mac": rogue, "req_rate": rate},
        }

    # NTP server crash
    if host == "ntp-server" and node.get("status") == "down":
        err = node.get("ntpd_error", "segfault")
        drift = node.get("clock_drift_s", 380)
        return {
            "output": (f"Apr 08 13:38:00 ntp-server ntpd[777]: {err}\n"
                       f"Apr 08 13:38:05 ntp-server kernel: ntpd[777]: segfault at 0 ip 0000 sp 0000\n"
                       f"Apr 08 13:39:00 ntp-server systemd[1]: ntpd.service: Failed\n"
                       f"Apr 08 13:44:00 ntp-server kernel: Clock drift detected: {drift}s"),
            "metadata": {"ntpd_crashed": True, "clock_drift_s": drift},
        }

    # Core router BGP logs
    if host == "core-router" and node.get("bgp_session") == "down":
        peer = node.get("bgp_peer", "10.1.0.254")
        reason = node.get("bgp_reset_reason", "hold_timer_expired")
        return {
            "output": (f"Apr 08 13:42:01 core-router bgpd[1234]: %BGP-5-ADJCHANGE: "
                       f"neighbor {peer} Down {reason.replace('_',' ').title()}\n"
                       f"Apr 08 13:42:01 core-router bgpd[1234]: Removing route 10.1.0.30/24 from RIB\n"
                       f"Apr 08 13:42:02 core-router bgpd[1234]: BGP session state: IDLE (was ESTABLISHED)\n"
                       f"Apr 08 13:42:02 core-router kernel: Route 10.1.0.30/24 withdrawn from FIB"),
            "metadata": {"bgp_down": True, "peer": peer, "reason": reason},
        }

    # App-server DB pool exhaustion
    if host == "app-server" and node.get("db_pool_exhausted"):
        return {
            "output": (f"Apr 08 13:43:15 app-server app[5678]: ERROR: connection pool timeout after 30s\n"
                       f"Apr 08 13:43:15 app-server app[5678]: No available connections to db-server:5432 "
                       f"({node.get('pool_used',100)}/{node.get('pool_size',100)} slots used)\n"
                       f"Apr 08 13:43:16 app-server app[5678]: FATAL: upstream db unreachable — returning 502"),
            "metadata": {"db_unreachable": True, "pool_exhausted": True},
        }

    # Split-brain cluster nodes
    if host.startswith("cluster-node") and node.get("split_brain"):
        return {
            "output": (f"Apr 08 14:01:00 {host} raft[42]: WARN: granted vote to self — starting election\n"
                       f"Apr 08 14:01:01 {host} raft[42]: Became leader (term 7) — heartbeat_timeout=100ms\n"
                       f"Apr 08 14:01:02 {host} raft[42]: WARN: existing leader (cluster-node-1) still active!\n"
                       f"Apr 08 14:01:03 {host} raft[42]: SPLIT-BRAIN detected — two leaders in term 7"),
            "metadata": {"split_brain": True, "role": "stale_leader", "heartbeat_timeout_ms": node.get("heartbeat_timeout_ms", 100)},
        }

    return {
        "output": f"Apr 08 13:44:00 {host} kernel: System operating normally. No errors.",
        "metadata": {"normal": True},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  FIREWALL / NETWORK POLICY
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_iptables",
    description="List iptables rules on a host (requires root / sudo).",
    os_support=LINUX_ONLY,
    parameters_schema={
        "host":  {"type": "string", "required": True},
        "chain": {"type": "string", "required": False, "default": "all",
                  "description": "INPUT | OUTPUT | FORWARD | all"},
    },
    cost_penalty=-0.15,
    category="firewall",
)
async def tool_check_iptables(params: Dict, graph: nx.DiGraph) -> Dict:
    host  = params.get("host", "")
    chain = params.get("chain", "all").upper()

    if host not in graph.nodes:
        return {"output": f"ssh: {host}: No route to host", "metadata": {}}

    # Check edges FROM host-a TO internet-router for firewall block
    for src, dst, attrs in graph.edges(data=True):
        if dst == host and attrs.get("firewall_blocked"):
            rule = attrs.get("iptables_rule", "-A FORWARD -j DROP")
            return {
                "output": (f"Chain INPUT (policy ACCEPT)\n"
                           f"target  prot opt source         destination\n\n"
                           f"Chain FORWARD (policy ACCEPT)\n"
                           f"target  prot opt source         destination\n"
                           f"DROP    all  --   10.0.0.5       0.0.0.0/0    *** SUSPICIOUS RULE ***\n"
                           f"  ({rule})\n\n"
                           f"Chain OUTPUT (policy ACCEPT)"),
                "metadata": {"firewall_blocked": True, "rule": rule},
            }

    return {
        "output": (f"Chain INPUT (policy ACCEPT)\ntarget  prot opt  -- 0.0.0.0/0  0.0.0.0/0\n\n"
                   f"Chain FORWARD (policy ACCEPT)\n\nChain OUTPUT (policy ACCEPT)\n"
                   f"  No DROP rules detected."),
        "metadata": {"firewall_blocked": False},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DHCP
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_dhcp",
    description="Inspect DHCP lease table and pool utilisation on the DHCP server.",
    os_support=LINUX_ONLY,
    parameters_schema={
        "host": {"type": "string", "required": True, "description": "DHCP server hostname"},
    },
    cost_penalty=-0.10,
    category="dhcp",
)
async def tool_check_dhcp(params: Dict, graph: nx.DiGraph) -> Dict:
    host = params.get("host", "")

    if host not in graph.nodes:
        return {"output": f"ssh: {host}: No route to host", "metadata": {}}

    node = graph.nodes[host]

    if node.get("pool_exhausted"):
        total    = node.get("leases_total", 254)
        used     = node.get("leases_used", 254)
        rogue    = node.get("rogue_mac", "de:ad:be:ef:00:01")
        rate     = node.get("rogue_req_rate", 1200)
        return {
            "output": (f"DHCP Pool: 10.0.0.0/24\n"
                       f"  Total leases : {total}\n"
                       f"  Active leases: {used}  ({used/total*100:.0f}%)\n"
                       f"  Free leases  : 0\n\n"
                       f"Top MAC by request rate:\n"
                       f"  {rogue}  {rate} req/min  <-- ANOMALOUS\n\n"
                       f"Status: POOL EXHAUSTED — new clients will not receive addresses"),
            "metadata": {
                "pool_exhausted": True,
                "used": used,
                "total": total,
                "rogue_mac": rogue,
                "rogue_req_rate": rate,
            },
        }

    return {
        "output": ("DHCP Pool: 10.0.0.0/24\n"
                   "  Total leases : 254\n"
                   "  Active leases: 42  (16%)\n"
                   "  Free leases  : 212\n\n"
                   "Status: OK"),
        "metadata": {"pool_exhausted": False, "used": 42, "total": 254},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NTP / TIME
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_ntp",
    description="Check NTP sync status and clock drift on a host.",
    os_support=LINUX_ONLY,
    parameters_schema={
        "host": {"type": "string", "required": True},
    },
    cost_penalty=-0.05,
    category="time",
)
async def tool_check_ntp(params: Dict, graph: nx.DiGraph) -> Dict:
    host = params.get("host", "")

    if host not in graph.nodes:
        return {"output": f"ssh: {host}: No route to host", "metadata": {}}

    node = graph.nodes[host]

    # NTP server itself is down
    if host == "ntp-server" and node.get("status") == "down":
        drift = node.get("clock_drift_s", 380)
        return {
            "output": (f"chronyc: Cannot talk to daemon — is chronyd/ntpd running?\n"
                       f"  ntpd.service: inactive (dead)\n"
                       f"  Estimated local clock drift: +{drift}s"),
            "metadata": {"ntp_running": False, "drift_s": drift},
        }

    # Clients affected by NTP drift
    if node.get("tls_errors") and node.get("clock_drift_s"):
        drift = node.get("clock_drift_s")
        return {
            "output": (f"chronyc tracking:\n"
                       f"  Reference ID : 0.0.0.0 (No sync source)\n"
                       f"  Stratum      : 16  (unsynchronised)\n"
                       f"  System time  : +{drift}.000 seconds fast of NTP time\n"
                       f"  Last offset  : +{drift}.000 seconds\n"
                       f"  Root dispersion: 999.999 seconds"),
            "metadata": {"ntp_synced": False, "drift_s": drift, "stratum": 16},
        }

    return {
        "output": ("chronyc tracking:\n"
                   "  Reference ID : 10.0.0.3 (ntp-server)\n"
                   "  Stratum      : 2\n"
                   "  System time  : +0.000123 seconds fast of NTP time\n"
                   "  RMS offset   : 0.000098 seconds"),
        "metadata": {"ntp_synced": True, "drift_s": 0.000123},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  BGP / ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_bgp",
    description="Show BGP peer status and routing table summary on a router.",
    os_support=LINUX_ONLY,
    parameters_schema={
        "host": {"type": "string", "required": True, "description": "Router hostname"},
    },
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
            "output": (f"BGP router identifier {node.get('ip','?')}, local AS number 65001\n"
                       f"BGP table version is 7, main routing table version 7\n\n"
                       f"Neighbor        V    AS MsgRcvd MsgSent   TblVer InQ OutQ Up/Down  State\n"
                       f"{peer:<15} 4 65002    1234    1235        7   0    0  00:00:03 {reason.upper()}\n\n"
                       f"Total number of neighbors 1\n"
                       f"WARNING: Route 10.1.0.30/24 removed from RIB (no valid path)"),
            "metadata": {
                "bgp_up": False,
                "peer": peer,
                "state": reason,
                "routes_withdrawn": ["10.1.0.30/24"],
            },
        }

    return {
        "output": (f"BGP router identifier {node.get('ip','?')}, local AS 65001\n"
                   f"Neighbor        V    AS MsgRcvd MsgSent   State\n"
                   f"10.1.0.254     4 65002   4523    4521  ESTABLISHED\n"
                   f"Total prefixes: 12"),
        "metadata": {"bgp_up": True},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLUSTER / DISTRIBUTED SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════════

@tool_registry.register(
    name="check_cluster",
    description="Inspect distributed cluster membership, leader status, and quorum.",
    os_support=LINUX_ONLY,
    parameters_schema={
        "host": {"type": "string", "required": True, "description": "Any cluster node"},
    },
    cost_penalty=-0.10,
    category="cluster",
)
async def tool_check_cluster(params: Dict, graph: nx.DiGraph) -> Dict:
    host = params.get("host", "")

    if host not in graph.nodes:
        return {"output": f"ssh: {host}: No route to host", "metadata": {}}

    # Collect all cluster nodes
    cluster_nodes = {
        n: d for n, d in graph.nodes(data=True) if n.startswith("cluster-node")
    }
    if not cluster_nodes:
        return {"output": f"No cluster nodes found in topology.", "metadata": {}}

    leaders    = [n for n, d in cluster_nodes.items() if d.get("cluster_role") == "leader"]
    split      = len(leaders) > 1
    lines      = [f"Cluster membership view from {host}:", ""]

    for name, attrs in sorted(cluster_nodes.items()):
        role = attrs.get("cluster_role", "unknown")
        hb   = attrs.get("heartbeat_timeout_ms", "?")
        flag = "  <-- STALE LEADER (SPLIT-BRAIN)" if attrs.get("split_brain") else ""
        lines.append(f"  {name} ({attrs.get('ip','?')})  role={role}  heartbeat_timeout={hb}ms{flag}")

    lines.append("")
    if split:
        lines.append(f"ALERT: SPLIT-BRAIN DETECTED — {len(leaders)} nodes claiming leadership: {leaders}")
        lines.append("  Recommended: fence stale leader, increase heartbeat_timeout to ≥300ms")
    else:
        lines.append(f"Quorum: OK  |  Leader: {leaders[0] if leaders else 'none'}")

    return {
        "output": "\n".join(lines),
        "metadata": {
            "split_brain": split,
            "leaders":     leaders,
            "node_count":  len(cluster_nodes),
        },
    }
