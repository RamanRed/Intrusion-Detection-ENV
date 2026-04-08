"""
network_environment.py — Advanced NetworkDiagnosticsEnvironment.

Improvements over v1:
  - Tracks tool_names list alongside cost sum (feeds diversity/quality scorers)
  - Max-step truncation with truncated=True on the final StepResult
  - Step timeout simulation: slow tools add simulated latency metadata
  - State() now returns full tool_call_log for richer observability
  - Thread-safe episode_id generation using uuid4
  - CallToolAction now passes os_profile for OS-filtered tool support checking
  - ResolveAction returns full reward breakdown in obs.info
"""

import time
import uuid
from typing import Optional, Literal, Union, List

try:
    from ..models import (
        Action, StepResult, EpisodeState,
        NetAction, ListToolsAction, CallToolAction, ResolveAction, NetObservation
    )
    from .scenario_generator import ScenarioGenerator
    from .reward_engine import RewardEngine
    from .tool_registry import tool_registry
except (ImportError, ModuleNotFoundError):
    from models import (
        Action, StepResult, EpisodeState,
        NetAction, ListToolsAction, CallToolAction, ResolveAction, NetObservation
    )
    from server.scenario_generator import ScenarioGenerator
    from server.reward_engine import RewardEngine
    from server.tool_registry import tool_registry


class Environment:
    async def reset(self, *args, **kwargs) -> StepResult:
        raise NotImplementedError

    async def step(self, action: Action) -> StepResult:
        raise NotImplementedError

    async def state(self) -> EpisodeState:
        raise NotImplementedError


class NetworkDiagnosticsEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self.scenario_generator = ScenarioGenerator()

        # Episode state
        self.graph             = None
        self.ground_truth      = None
        self.episode_id        = str(uuid.uuid4())
        self.step_count        = 0
        self.cumulative_reward = 0.0
        self.start_time        = 0.0
        self.tool_names:  List[str] = []
        self.tool_cost_sum         = 0.0
        self.os_profile            = "linux"
        self.scenario_id           = "dns_failure"
        self.difficulty            = "medium"
        self.max_steps             = 20
        self.done                  = False
        self.reward_engine: Optional[RewardEngine] = None

    # ── Reset ──────────────────────────────────────────────────────────────────

    async def reset(
        self,
        seed:                  Optional[int]  = None,
        os_profile:            Literal['linux', 'windows', 'macos', 'android'] = 'linux',
        scenario_id:           Optional[str]  = None,
        difficulty:            Literal['easy', 'medium', 'hard', 'expert'] = 'medium',
        multi_device:          bool           = False,
        partial_observability: float          = 0.8,
    ) -> StepResult[NetObservation]:

        self.os_profile    = os_profile
        self.scenario_id   = scenario_id or "dns_failure"
        self.difficulty    = difficulty
        self.step_count    = 0
        self.cumulative_reward = 0.0
        self.start_time    = time.time()
        self.tool_names    = []
        self.tool_cost_sum = 0.0
        self.done          = False
        self.episode_id    = str(uuid.uuid4())

        # Max steps from scenario registry; fall back to a default
        from server.scenario_generator import TASK_MAP  # noqa: late import avoids circular
        task = TASK_MAP.get(self.scenario_id, {})
        self.max_steps = task.get("max_steps", 20)

        self.reward_engine = RewardEngine(difficulty)

        self.graph, self.ground_truth = self.scenario_generator.generate(
            scenario_id=self.scenario_id,
            os_profile=os_profile,
            difficulty=difficulty,
            seed=seed,
            partial_observability=partial_observability,
        )

        obs = NetObservation(
            stdout=(
                f"[NetworkDiagnosticsEnv] Episode {self.episode_id[:8]} started.\n"
                f"Scenario : {self.scenario_id}  |  Difficulty: {difficulty.upper()}\n"
                f"OS       : {os_profile}\n"
                f"Max steps: {self.max_steps}\n"
                f"Use ListToolsAction to discover available diagnostic tools."
            ),
            done=False,
            reward=0.0,
            info={
                "episode_id":   self.episode_id,
                "scenario_id":  self.scenario_id,
                "difficulty":   difficulty,
                "max_steps":    self.max_steps,
            },
        )
        return StepResult(observation=obs, reward=0.0, done=False, truncated=False,
                          info={"episode_id": self.episode_id})

    # ── Step ───────────────────────────────────────────────────────────────────

    async def step(
        self,
        action: Union[NetAction, ListToolsAction, CallToolAction, ResolveAction],
    ) -> StepResult[NetObservation]:

        if self.done:
            obs = NetObservation(
                stdout="Episode already finished. Call /reset to start a new episode.",
                done=True, reward=0.0,
            )
            return StepResult(observation=obs, reward=0.0, done=True, truncated=False)

        self.step_count += 1
        obs         = NetObservation(stdout="", done=False, reward=0.0)
        reward_step = 0.0
        t_start     = time.time()

        # ── Dispatch on action type ────────────────────────────────────────────

        if isinstance(action, ListToolsAction):
            all_tools = tool_registry.list_tools()
            # Filter by OS support
            os_tools = [
                t for t in all_tools
                if self.os_profile in t.get("os_support", [self.os_profile])
            ] if hasattr(all_tools[0], "get") else all_tools

            obs.available_tools = all_tools
            obs.stdout = (
                f"Available tools ({len(all_tools)} total):\n" +
                "\n".join(
                    f"  [{t['category'].upper():12s}] {t['name']:20s} — {t['description']}  "
                    f"(cost={t['cost_penalty']})"
                    for t in all_tools
                )
            )
            self.tool_names.append("ListToolsAction")

        elif isinstance(action, CallToolAction):
            tool = tool_registry.get_tool(action.tool_name)
            if not tool:
                obs.stderr    = (f"Tool '{action.tool_name}' not found. "
                                 f"Use ListToolsAction to discover available tools.")
                reward_step   = -0.2   # light penalty for invalid tool

            elif self.os_profile not in tool.get("os_support", [self.os_profile]):
                obs.stderr    = (f"Tool '{action.tool_name}' is not supported on OS "
                                 f"'{self.os_profile}'. "
                                 f"Supported: {tool['os_support']}")
                reward_step   = -0.1

            else:
                self.tool_names.append(action.tool_name)
                self.tool_cost_sum += tool["cost_penalty"]
                reward_step         = tool["cost_penalty"]

                result = await tool["handler"](action.parameters, self.graph)
                obs.tool_result = result
                obs.stdout      = result.get("output", "Tool executed (no output).")
                # Embed structured metadata in info
                obs.info        = {"tool_metadata": result.get("metadata", {})}

        elif isinstance(action, NetAction):
            # Generic shell command — mild reward, no structured output
            cmd = action.command.strip()
            obs.stdout = (
                f"$ {cmd}\n"
                f"(Generic NetAction executed. "
                f"For structured results and tool metadata, prefer CallToolAction.)"
            )
            self.tool_names.append(f"NetAction:{cmd[:30]}")

        elif isinstance(action, ResolveAction):
            self.done = True
            obs.done  = True

            breakdown = self.reward_engine.compute_breakdown(
                target_root_cause=self.ground_truth.get("root_cause", ""),
                claimed_cause=action.root_cause,
                is_resolved=True,
                tools_called=self.step_count,
                tool_cost_sum=self.tool_cost_sum,
                tool_names=self.tool_names,
                max_steps=self.max_steps,
            )
            reward_step  = breakdown["total"]
            obs.stdout   = (
                f"[RESOLVED] Root cause submitted: '{action.root_cause}'\n"
                f"Fix applied : '{action.fix_applied}'\n"
                f"Score       : {breakdown['total']:.4f}  |  Passed: {breakdown['passed']}\n"
                f"RCA accuracy: {breakdown['dimensions']['rca_accuracy']:.4f}\n"
                f"Efficiency  : {breakdown['dimensions']['efficiency']:.4f}\n"
                f"Tool economy: {breakdown['dimensions']['tool_economy']:.4f}"
            )
            obs.info = {"reward_breakdown": breakdown}

        else:
            obs.stderr  = f"Unrecognised action type: {type(action).__name__}"
            reward_step = -0.1

        # ── Bookkeeping ────────────────────────────────────────────────────────
        self.cumulative_reward += reward_step
        obs.reward = reward_step
        obs.done   = self.done

        # Step timing (useful for latency-aware agents)
        elapsed_ms = round((time.time() - t_start) * 1000, 1)

        # Max-step truncation
        truncated = False
        if self.step_count >= self.max_steps and not self.done:
            truncated    = True
            self.done    = True
            obs.done     = True
            obs.stdout  += (f"\n[TRUNCATED] Maximum steps ({self.max_steps}) reached "
                            f"without a ResolveAction. Episode ended.")

        return StepResult(
            observation=obs,
            reward=reward_step,
            done=self.done,
            truncated=truncated,
            info={"step": self.step_count, "elapsed_ms": elapsed_ms},
        )

    # ── State ──────────────────────────────────────────────────────────────────

    async def state(self) -> EpisodeState:
        return EpisodeState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            cumulative_reward=round(self.cumulative_reward, 4),
            time_elapsed_s=round(time.time() - self.start_time, 2),
            scenario_id=self.scenario_id,
            os_profile=self.os_profile,
            done=self.done,
            tools_called=list(self.tool_names),   # full ordered log
        )
