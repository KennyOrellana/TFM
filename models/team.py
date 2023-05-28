import copy

from pydantic import BaseModel


class Team(BaseModel):
    id: str
    agents = [str]
    k_robustness = 0

    def get_cost(self):
        return sum([agent.get_cost() for agent in self.agents])

    # is said to be c-costly if the cost of T is less than c
    def is_affordable(self, c):
        return self.get_cost() <= c

    # is said to be efficient with respect to G if T can accomplish G
    def is_efficient(self, goal):
        return all([goal.can_complete_task(agent) for agent in self.agents])

    def get_env_arguments(self, env_params):
        env_arguments = copy.deepcopy(env_params)
        env_arguments["n_agents"] = len(self.agents)

        return env_arguments
