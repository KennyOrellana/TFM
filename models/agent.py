# a class for Agent which contain a lisk of task that can execute, also the cost of the agent
from pydantic import BaseModel


class Agent(BaseModel):
    id: str
    skills: list[str]
    cost = int

    # fun_cost = fun_cost
    # def __init__(self, skills, cost=0, fun_cost=None):
    #     self.skills = skills  # A skill can do a specific task or a sub-set of tasks
    #     self._cost = cost  # The cost of using this agent
    #     self.fun_cost = fun_cost

    # def get_cost(self):
    #     if self.fun_cost is None:
    #         return self._cost
    #     else:
    #         return self.fun_cost(self)

    def get_cost(self):
        return self.cost
