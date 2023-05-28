from pydantic import BaseModel


class Settings(BaseModel):
    name: str
    render: bool = False
    save: bool = False
    failure_probability: float = 0.0
