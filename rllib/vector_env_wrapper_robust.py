import random
from typing import List

import numpy as np
import torch
from vmas.simulator.environment.environment import Environment
from vmas.simulator.environment.rllib import VectorEnvWrapper


class VectorEnvWrapperRobust(VectorEnvWrapper):
    def __init__(
            self,
            env: Environment,
            k_robustness: int = 0,
            failure_probability: float = 0.0,
    ):
        self.k_robustness = k_robustness
        self.failure_probability_per_step = failure_probability / (env.max_steps * env.n_agents)
        self._initialize_agents(env.num_envs)
        super().__init__(env=env)

    def _initialize_agents(self, num_envs: int):
        self.failed_agents = np.empty(num_envs, dtype=object)
        for i in range(num_envs):
            self.failed_agents[i] = np.array([], dtype=int)

    def _compute_failed_agents(self):
        for env_index in range(self.num_envs):  # Compute for all environments
            if len(self.failed_agents[env_index]) < self.k_robustness:  # If not enough agents failed
                random_probability = random.random()
                if random_probability <= self.failure_probability_per_step:  # Agents fails based on probability
                    print(
                        f"[VectorEnvWrapperRobust] Random probability: {random_probability} <= {self.failure_probability_per_step}")
                    failed_agent_index = random.choice(range(self.env.n_agents - 1))  # Choose random agent
                    if failed_agent_index not in self.failed_agents[env_index]:  # If agent not already failed, add it
                        self.failed_agents[env_index] = np.append(self.failed_agents[env_index], failed_agent_index)

    def _action_list_to_tensor(self, list_in: List) -> List:
        self._compute_failed_agents()
        return self._action_list_to_tensor_robust(list_in)

    def _action_list_to_tensor_robust(self, list_in: List) -> List:
        if len(list_in) == self.num_envs:
            actions = []
            for agent in self._env.agents:
                actions.append(
                    torch.zeros(
                        self.num_envs,
                        self._env.get_agent_action_size(agent),
                        device=self._env.device,
                        dtype=torch.float32,
                    )
                )
            for j in range(self.num_envs):
                assert (
                        len(list_in[j]) == self._env.n_agents
                ), f"Expecting actions for {self._env.n_agents} agents, got {len(list_in[j])} actions"
                for i in range(self._env.n_agents):
                    # Here start the changes from base class
                    # Skip actions/movements of failed agents
                    if i in self.failed_agents[j]:
                        continue
                    # Here end the changes from base class

                    act = torch.tensor(
                        list_in[j][i], dtype=torch.float32, device=self._env.device
                    )
                    if len(act.shape) == 0:
                        assert (
                                self._env.get_agent_action_size(self._env.agents[i]) == 1
                        ), f"Action of agent {i} in env {j} is supposed to be an scalar int"
                    else:
                        assert len(act.shape) == 1 and act.shape[
                            0
                        ] == self._env.get_agent_action_size(self._env.agents[i]), (
                            f"Action of agent {i} in env {j} hase wrong shape: "
                            f"expected {self._env.get_agent_action_size(self._env.agents[i])}, got {act.shape[0]}"
                        )
                    actions[i][j] = act
            return actions
        else:
            assert False, "Input action is not in correct format"
