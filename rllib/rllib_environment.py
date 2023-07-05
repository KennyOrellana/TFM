import os
from typing import Dict, Optional

import numpy as np
import ray
import wandb
from ray import tune
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.rllib.evaluation import Episode, MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env
from ray.tune.integration.wandb import WandbLoggerCallback
from vmas import make_env, Wrapper
from vmas.examples.rllib import RenderingCallbacks, EvaluationCallbacks

from rllib.vector_env_wrapper_robust import VectorEnvWrapperRobust

supported_environments = ["balance", "ball_trajectory", "discovery", "dispersion"]


def env_creator_robust(config: Dict):
    env = make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        max_steps=config["max_steps"],
        # Scenario specific variables
        **config["scenario_config"],
    )
    scenario_config = config["scenario_config"]
    return VectorEnvWrapperRobust(env, scenario_config["k_robustness"], scenario_config["failure_probability"])


class RLlibEnvironment:
    def __init__(self, scenario_name, n_agents: int, k_robustness: int = 0, failure_probability: float = 0.0):
        self.n_agents = n_agents
        self.num_vectorized_envs = 96
        self.num_workers = 5
        self.vmas_device = "cpu"
        self.continuous_actions = True
        self.max_steps = 200
        self.scenario_config = {}
        self.scenario_name = scenario_name
        self.k_robustness = k_robustness
        self.failure_probability = failure_probability

        self.initialize()
        self.train()

    def initialize(self):
        if not ray.is_initialized():
            ray.init()
            print("Ray init robust!")
        register_env(f"tfm_{self.scenario_name}", lambda config: env_creator_robust(config))

    class EvaluationCallbacks(DefaultCallbacks):
        def on_episode_step(
                self,
                *,
                worker: RolloutWorker,
                base_env: BaseEnv,
                episode: MultiAgentEpisode,
                **kwargs,
        ):
            info = episode.last_info_for()
            for a_key in info.keys():
                for b_key in info[a_key]:
                    try:
                        episode.user_data[f"{a_key}/{b_key}"].append(info[a_key][b_key])
                    except KeyError:
                        episode.user_data[f"{a_key}/{b_key}"] = [info[a_key][b_key]]

        def on_episode_end(
                self,
                *,
                worker: RolloutWorker,
                base_env: BaseEnv,
                policies: Dict[str, Policy],
                episode: MultiAgentEpisode,
                **kwargs,
        ):
            info = episode.last_info_for()
            for a_key in info.keys():
                for b_key in info[a_key]:
                    metric = np.array(episode.user_data[f"{a_key}/{b_key}"])
                    episode.custom_metrics[f"{a_key}/{b_key}"] = np.sum(metric).item()

    class RenderingCallbacks(DefaultCallbacks):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.frames = []

        def on_episode_step(
                self,
                *,
                worker: RolloutWorker,
                base_env: BaseEnv,
                policies: Optional[Dict[PolicyID, Policy]] = None,
                episode: Episode,
                **kwargs,
        ) -> None:
            self.frames.append(base_env.vector_env.try_render_at(mode="rgb_array"))

        def on_episode_end(
                self,
                *,
                worker: RolloutWorker,
                base_env: BaseEnv,
                policies: Dict[PolicyID, Policy],
                episode: Episode,
                **kwargs,
        ) -> None:
            vid = np.transpose(self.frames, (0, 3, 1, 2))
            episode.media["rendering"] = wandb.Video(
                vid, fps=1 / base_env.vector_env.env.world.dt, format="mp4"
            )
            self.frames = []

    def train(self):
        RLLIB_NUM_GPUS = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        num_gpus = 0.001 if RLLIB_NUM_GPUS > 0 else 0  # Driver GPU
        num_gpus_per_worker = (
            (RLLIB_NUM_GPUS - num_gpus) / (self.num_workers + 1) if self.vmas_device == "cuda" else 0
        )

        tune.run(
            PPOTrainer,
            stop={"training_iteration": 5000},
            checkpoint_freq=1,
            keep_checkpoints_num=2,
            checkpoint_at_end=True,
            checkpoint_score_attr="episode_reward_mean",
            callbacks=[
                WandbLoggerCallback(
                    project=f"{self.scenario_name}",
                    api_key="",
                )
            ],
            config={
                "seed": 0,
                "framework": "torch",
                "env": f"tfm_{self.scenario_name}",
                "kl_coeff": 0.01,
                "kl_target": 0.01,
                "lambda": 0.9,
                "clip_param": 0.2,
                "vf_loss_coeff": 1,
                "vf_clip_param": float("inf"),
                "entropy_coeff": 0,
                "train_batch_size": 60000,
                "rollout_fragment_length": 125,
                # "sgd_minibatch_size": 10,
                "sgd_minibatch_size": 4096,
                "num_sgd_iter": 40,
                "num_gpus": num_gpus,
                "num_workers": self.num_workers,
                "num_gpus_per_worker": num_gpus_per_worker,
                "num_envs_per_worker": self.num_vectorized_envs,
                "lr": 5e-5,
                "gamma": 0.99,
                "use_gae": True,
                "use_critic": True,
                "batch_mode": "truncate_episodes",
                "env_config": {
                    "device": self.vmas_device,
                    "num_envs": self.num_vectorized_envs,
                    "scenario_name": self.scenario_name,
                    "continuous_actions": self.continuous_actions,
                    "max_steps": self.max_steps,
                    # Scenario specific variables
                    "scenario_config": {
                        "n_agents": self.n_agents,
                        "k_robustness": self.k_robustness,
                        "failure_probability": self.failure_probability,
                    },
                },
                "evaluation_interval": 5,
                "evaluation_duration": 1,
                "evaluation_num_workers": 1,
                "evaluation_parallel_to_training": True,
                "evaluation_config": {
                    "num_envs_per_worker": 1,
                    "env_config": {
                        "num_envs": 1,
                    },
                    "callbacks": MultiCallbacks([RenderingCallbacks, EvaluationCallbacks]),
                },
                "callbacks": EvaluationCallbacks,
            },
        )
