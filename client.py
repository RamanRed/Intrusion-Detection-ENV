from typing import Optional, Literal, Union

from .models import (
    StepResult, EpisodeState,
    NetAction, ListToolsAction, CallToolAction, ResolveAction, NetObservation
)


# Inline base class so my_env/ is self-contained in Docker
class EnvClient:
    def __init__(self, base_url: str, **kwargs):
        self.base_url = base_url

    async def reset(self, *args, **kwargs) -> StepResult:
        raise NotImplementedError

    async def step(self, action) -> StepResult:
        raise NotImplementedError

    async def state(self) -> EpisodeState:
        raise NotImplementedError


class NetOSDiagEnv(EnvClient):
    def __init__(
        self,
        base_url: str = "ws://localhost:7860",
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        max_message_size_mb: float = 100.0,
        mode: Literal['simulation', 'production'] = 'simulation'
    ):
        super().__init__(base_url=base_url)
        self.mode = mode
        self._server_env = None

    def set_local_server(self, server_env):
        self._server_env = server_env

    async def reset(
        self,
        seed: Optional[int] = None,
        os_profile: Literal['linux', 'windows', 'macos', 'android'] = 'linux',
        scenario_id: Optional[str] = None,
        difficulty: Literal['easy', 'medium', 'hard', 'expert'] = 'medium',
        multi_device: bool = False,
        partial_observability: float = 0.8,
        enable_nika: bool = True,
        enable_synthetic: bool = True
    ) -> StepResult[NetObservation]:

        if self._server_env:
            return await self._server_env.reset(
                seed=seed, os_profile=os_profile, scenario_id=scenario_id,
                difficulty=difficulty, multi_device=multi_device,
                partial_observability=partial_observability
            )
        raise NotImplementedError("WebSocket connection not implemented in this mock client.")

    async def step(
        self,
        action: Union[NetAction, ListToolsAction, CallToolAction, ResolveAction]
    ) -> StepResult[NetObservation]:

        if self._server_env:
            return await self._server_env.step(action)
        raise NotImplementedError("WebSocket connection not implemented in this mock client.")

    async def state(self) -> EpisodeState:
        if self._server_env:
            return await self._server_env.state()
        raise NotImplementedError("WebSocket connection not implemented in this mock client.")
