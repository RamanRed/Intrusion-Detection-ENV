import time
from typing import Optional, Literal, Union

# Support both package-relative (local dev) and absolute (Docker/top-level) imports
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


# Inline base class so Docker needs no external openenv_core
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
        self.graph = None
        self.ground_truth = None
        self.episode_id = "test_ep_01"
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.start_time = 0
        self.tools_called = []
        self.tool_cost_sum = 0.0
        self.os_profile = "linux"
        self.scenario_id = "default"
        self.done = False

    async def reset(
        self,
        seed: Optional[int] = None,
        os_profile: Literal['linux', 'windows', 'macos', 'android'] = 'linux',
        scenario_id: Optional[str] = None,
        difficulty: Literal['easy', 'medium', 'hard', 'expert'] = 'medium',
        multi_device: bool = False,
        partial_observability: float = 0.8
    ) -> StepResult[NetObservation]:

        self.os_profile = os_profile
        self.scenario_id = scenario_id or "dns_failure"
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.start_time = time.time()
        self.tools_called = []
        self.tool_cost_sum = 0.0
        self.done = False

        self.reward_engine = RewardEngine(difficulty)

        self.graph, self.ground_truth = self.scenario_generator.generate(
            self.scenario_id, self.os_profile, difficulty
        )

        obs = NetObservation(
            stdout="Environment ready. Type a command or use ListToolsAction.",
            done=False,
            reward=0.0
        )
        return StepResult(observation=obs, reward=0.0, done=False, truncated=False)

    async def step(self, action: Union[NetAction, ListToolsAction, CallToolAction, ResolveAction]) -> StepResult[NetObservation]:
        self.step_count += 1

        obs = NetObservation(stdout="", done=False, reward=0.0)
        reward_step = 0.0

        if isinstance(action, ListToolsAction):
            obs.available_tools = tool_registry.list_tools()
            obs.stdout = "Tools discovered."
            self.tools_called.append("ListToolsAction")

        elif isinstance(action, CallToolAction):
            tool = tool_registry.get_tool(action.tool_name)
            if not tool:
                obs.stderr = f"Command {action.tool_name} not found."
                reward_step = -0.5
            else:
                self.tools_called.append(action.tool_name)
                self.tool_cost_sum += tool["cost_penalty"]
                reward_step += tool["cost_penalty"]

                handler = tool["handler"]
                result = await handler(action.parameters, self.graph)
                obs.tool_result = result
                obs.stdout = result.get("output", "Tool executed.")

        elif isinstance(action, NetAction):
            obs.stdout = f"Executed generic shell command: {action.command}. (Use CallToolAction for tool calls)"
            self.tools_called.append("NetAction raw")

        elif isinstance(action, ResolveAction):
            self.done = True
            obs.done = True

            acc_reward = self.reward_engine.compute(
                self.ground_truth["root_cause"],
                action.root_cause,
                is_resolved=True,
                tools_called=self.step_count,
                tool_cost_sum=self.tool_cost_sum
            )
            reward_step = acc_reward
            obs.stdout = f"Resolution submitted. Score: {acc_reward:.2f}"

        else:
            obs.stderr = "Unknown action type."

        self.cumulative_reward += reward_step
        obs.reward = reward_step
        obs.done = self.done

        return StepResult(observation=obs, reward=reward_step, done=self.done, truncated=False)

    async def state(self) -> EpisodeState:
        return EpisodeState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            cumulative_reward=self.cumulative_reward,
            time_elapsed_s=time.time() - self.start_time,
            scenario_id=self.scenario_id,
            os_profile=self.os_profile,
            done=self.done,
            tools_called=self.tools_called
        )
