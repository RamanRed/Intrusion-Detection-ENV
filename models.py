from dataclasses import dataclass, field
from typing import Generic, TypeVar, Literal, Dict, Any, List, Optional

T = TypeVar('T')

# ---------------------------------------------------------------------------
# Base classes (inlined to keep my_env/ self-contained in Docker)
# v2 - removed openenv_core dependency
# ---------------------------------------------------------------------------

@dataclass
class Action:
    pass

@dataclass
class Observation:
    pass

@dataclass
class StepResult(Generic[T]):
    observation: T
    reward: float
    done: bool
    truncated: bool
    info: Dict = field(default_factory=dict)

@dataclass
class EpisodeState:
    episode_id: str
    step_count: int
    cumulative_reward: float
    time_elapsed_s: float
    scenario_id: str
    os_profile: str
    done: bool
    tools_called: List[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Environment-specific models
# ---------------------------------------------------------------------------

OS_TYPE = Literal["linux", "windows", "macos", "android"]

@dataclass
class NetAction(Action):
    """Primary action: either raw command OR tool call"""
    command: str = ""
    tool_name: str = ""
    tool_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ListToolsAction(Action):
    """Agent asks: what tools are available right now?"""
    pass

@dataclass
class CallToolAction(Action):
    """Agent calls a discovered tool"""
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResolveAction(Action):
    """Agent claims issue is resolved"""
    root_cause: str
    fix_applied: str

@dataclass
class NetObservation(Observation):
    stdout: str
    stderr: str = ""
    exit_code: int = 0
    system_summary: Dict = field(default_factory=dict)
    available_tools: List[Dict] = field(default_factory=list)
    tool_result: Dict = field(default_factory=dict)
    done: bool = False
    reward: float = 0.0
    info: Dict = field(default_factory=dict)
