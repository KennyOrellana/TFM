from pydantic import BaseModel

from models.task import Task


class Goal(BaseModel):
    id: str
    tasks: list[Task]

    def can_complete_task(self, agent):
        return any([task.can_complete(agent.skills) for task in self.tasks])
