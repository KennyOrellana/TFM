from pydantic import BaseModel
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.scenarios.transport import HeuristicPolicy as TransportPolicy


# class Environment(ABC):
class Settings(BaseModel):
    name: str
    render: bool = False
    save: bool = False

