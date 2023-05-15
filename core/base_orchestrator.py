import copy
import json
from abc import ABC

from environments.base_environment import BaseEnvironment
from models.simulation import Simulation


class BaseOrchestrator(ABC):

    def __init__(self):
        self.simulation_data = None

    def execute(self, data_path='example/demo.json'):
        self._prepare(data_path)
        self._check_data()
        self._run()

    def on_error(self, error):
        pass

    def _prepare(self, data_path='example/demo.json'):
        with open(data_path, 'r') as json_file:
            data_dict = json.load(json_file)

        self.simulation_data = Simulation(**data_dict)

    def _check_data(self):
        assert self.simulation_data is not None
        assert self.simulation_data.environment is not None
        assert self.simulation_data.teams is not None
        assert self.simulation_data.goals is not None
        assert len(self.simulation_data.teams) > 0
        assert len(self.simulation_data.goals) > 0

        for goal in self.simulation_data.goals:
            assert len(goal.tasks) > 0
            assert len(goal.tasks) > 0

    def _run(self):
        for goal in self.simulation_data.goals:
            for task in goal.tasks:
                for team in self.simulation_data.teams:
                    agents = self.simulation_data.get_agents_of_team(team.id)
                    agents_can_complete_task = [agent for agent in agents if task.can_complete(agent.skills)]

                    env_arguments = copy.deepcopy(task.env_kwargs)
                    env_arguments["n_agents"] = len(agents_can_complete_task)
                    assert env_arguments["n_agents"] > 0

                    BaseEnvironment(task.get_scenario_name(), env_arguments)
