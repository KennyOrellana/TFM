import copy
import json

from core.run_heuristic import run_heuristic
from models import agent
from models.simulation import Simulation


def load_data(data_path='example/demo.json'):
    with open(data_path, 'r') as json_file:
        data_dict = json.load(json_file)

    simulation_data = Simulation(**data_dict)

    for goal in simulation_data.goals:
        for task in goal.tasks:
            for team in simulation_data.teams:
                agents = simulation_data.get_agents_of_team(team.id)
                agents_can_complete_task = [agent for agent in agents if task.can_complete(agent.skills)]

                env_arguments = copy.deepcopy(task.env_kwargs)

                n_agents = len(agents_can_complete_task)
                if n_agents > 1:
                    env_arguments["n_agents"] = n_agents
                else:
                    assert n_agents > 1



                run_heuristic(
                    scenario_name=task.get_scenario_name(),
                    heuristic=task.get_policy(),
                    n_steps=500,
                    n_envs=300,
                    env_kwargs=env_arguments,
                    render=simulation_data.environment.render,
                    save_render=simulation_data.environment.save,
                )


if __name__ == '__main__':
    load_data()
