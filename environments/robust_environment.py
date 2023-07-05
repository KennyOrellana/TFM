import random

from environments.base_environment import BaseEnvironment
from rllib.rllib_environment import RLlibEnvironment


class RobustEnvironment(BaseEnvironment):
    def __init__(self, name, n_agents, kwargs, settings, k_robustness):
        self.settings = settings
        self.agents_status = [True] * n_agents
        self.k_robustness = k_robustness
        super().__init__(name, n_agents, kwargs)

    def is_agent_active(self, agent_index):
        # print(self.agents_status, sum(self.agents_status), agent_index)
        if 0 < (self.n_agents - self.k_robustness) < sum(self.agents_status) and self.agents_status[agent_index]:
            self.agents_status[agent_index] = self.probability_of_failure()

        return self.agents_status[agent_index]

    def probability_of_failure(self):
        probability = self.settings.failure_probability
        return random.choices([True, False], weights=[10 - probability, probability])[0]

    def _initialize_rllib(self):
        return RLlibEnvironment(self.name, self.n_agents, self.k_robustness, self.settings.failure_probability)
