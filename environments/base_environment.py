import time
import torch
from abc import ABC

from vmas import make_env
from vmas.simulator.utils import save_video

from core.policy_provider import PolicyProvider
from environments.settings import Settings


class BaseEnvironment(ABC):
    def __init__(self, name, kwargs):
        self.name = name
        policy = PolicyProvider.get_policy_for(name)
        self.policy = policy(continuous_action=Settings.CONTINUOUS_ACTIONS)
        self.kwargs = kwargs
        self.steps = Settings.NUM_STEPS
        self.n_envs = Settings.NUM_ENVS
        self.render = True
        self.save_render = True
        self.env = self._initialize_environment()
        self._run()

    def _initialize_environment(self):
        return make_env(
            scenario=self.name,
            num_envs=self.n_envs,
            device=Settings.DEVICE,
            continuous_actions=Settings.CONTINUOUS_ACTIONS,
            wrapper=Settings.WRAPPER,
            **self.kwargs)

    def _run(self):
        frame_list = []  # For creating a gif
        init_time = time.time()
        step = 0
        obs = self.env.reset()
        total_reward = 0
        for s in range(self.steps):
            step += 1
            actions = [None] * len(obs)
            for i in range(len(obs)):
                actions[i] = self.policy.compute_action(obs[i], u_range=self.env.agents[i].u_range)
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
