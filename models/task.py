from pydantic import BaseModel


class Task(BaseModel):
    environment: str
    env_kwargs: dict = {}

    def can_complete(self, agent_skills):
        return self.environment in agent_skills

    def get_scenario_name(self) -> str:
        return self.environment  # TODO: raise exception if not supported
