from pydantic import BaseModel

from models.settings import Settings
from models.agent import Agent
from models.goal import Goal
from models.team import Team


class Simulation(BaseModel):
    environment: Settings
    agents: list[Agent]
    teams: list[Team]
    goals: list[Goal]

    def get_agents_of_team(self, team_id):
        agents_ids = [agent_id for team in self.teams if team.id == team_id for agent_id in team.agents]

        agents = []
        for agent_id in agents_ids:
            for agent in self.agents:
                if agent.id == agent_id:
                    agents.append(agent)

        return agents
