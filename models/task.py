from typing import Type

from pydantic import BaseModel
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.scenarios.transport import HeuristicPolicy as TransportPolicy, HeuristicPolicy
from vmas.scenarios.balance import HeuristicPolicy as BalancePolicy
from vmas.scenarios.wheel import HeuristicPolicy as WheelPolicy


class Task(BaseModel):
    environment: str
    env_kwargs: dict = {}

    def can_complete(self, agent_skills):
        return self.environment in agent_skills

    def get_scenario_name(self) -> str:
        return self.environment  # TODO: raise exception if not supported

    # @abstractmethod
    def get_policy(self) -> Type[HeuristicPolicy]:
        if self.environment == "transport":
            return TransportPolicy
        if self.environment == "balance":
            return BalancePolicy
        if self.environment == "wheel":
            return WheelPolicy
        else:
            return None  # TODO: raise exception
