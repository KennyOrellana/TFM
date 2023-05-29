import time
import torch
from abc import ABC

from vmas import make_env
from vmas.simulator.utils import save_video

from core.policy_provider import PolicyProvider
from environments.settings import Settings
from rllib.balance import BalanceEnvironment
from rllib.ball_trajectory import BallTrajectoryEnvironment


class BaseEnvironment(ABC):
    def __init__(self, name, n_agents, kwargs):
        self.name = name
        policy = PolicyProvider.get_policy_for(name)
        if policy is not None:
            self.policy = policy(continuous_action=Settings.CONTINUOUS_ACTIONS)
        self.n_agents = n_agents
        self.kwargs = kwargs
        self.steps = Settings.NUM_STEPS
        self.n_envs = Settings.NUM_ENVS
        self.render = True
        self.save_render = True
        self.env = self._initialize_environment()
        # self._run()

    def _initialize_environment(self):
        if self.name == "balance":
            return self._initialize_balance()
        elif self.name == "ball_trajectory":
            return self._initialize_ball_trajectory()
        else:
            return make_env(
                scenario=self.name,
                n_agents=self.n_agents,
                num_envs=self.n_envs,
                device=Settings.DEVICE,
                continuous_actions=Settings.CONTINUOUS_ACTIONS,
                wrapper=Settings.WRAPPER,
                random_package_pos_on_line=True,
                control_two_agents=True,
                **self.kwargs)

    def _run(self):
        if self.name == "balance":
            return

        frame_list = []  # For creating a gif
        init_time = time.time()
        step = 0
        obs = self.env.reset()
        total_reward = 0
        for s in range(self.steps):
            step += 1
            actions = [None] * len(obs)
            for i in range(len(obs)):
                if self.is_agent_active(i):
                    actions[i] = self.policy.compute_action(obs[i], u_range=self.env.agents[i].u_range)
                else:
                    actions[i] = self.policy.compute_action(obs[i], u_range=0.0)
            obs, rews, dones, info = self.env.step(actions)
            rewards = torch.stack(rews, dim=1)
            global_reward = rewards.mean(dim=1)
            mean_global_reward = global_reward.mean(dim=0)
            total_reward += mean_global_reward

            if dones.all():
                print("All elements are True")

            if self.render:
                frame_list.append(
                    self.env.render(
                        mode="rgb_array",
                        agent_index_focus=None,
                        visualize_when_rgb=True,
                    )
                )

        total_time = time.time() - init_time
        if self.render and self.save_render:
            save_video(self.name, frame_list, 1 / self.env.scenario.world.dt)

        print(
            f"It took: {total_time}s for {self.steps} steps of {self.n_envs} parallel environments\n"
            f"The average total reward was {total_reward}"
        )

    def _initialize_balance(self):
        return BalanceEnvironment(self.n_agents)

    def _initialize_ball_trajectory(self):
        return BallTrajectoryEnvironment(self.n_agents)

    def is_agent_active(self, agent_index):
        return True
