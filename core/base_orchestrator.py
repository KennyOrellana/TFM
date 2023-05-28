import copy
import json
from abc import ABC

from environments.base_environment import BaseEnvironment
from environments.robust_environment import RobustEnvironment
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
                    n_agents = len(agents_can_complete_task)
                    assert n_agents > 0

                    # Revisar porque hay un out of index, quizás debamos probar si cada team puede ejecutar cada tarea en lugar de como está ahorita
                    if team.k_robustness == 0:
                        BaseEnvironment(
                            name=task.get_scenario_name(),
                            n_agents=n_agents,
                            kwargs=env_arguments,
                        )
                    else:
                        RobustEnvironment(
                            name=task.get_scenario_name(),
                            n_agents=n_agents,
                            kwargs=env_arguments,
                            settings=self.simulation_data.environment,
                            k_robustness=team.k_robustness
                        )
