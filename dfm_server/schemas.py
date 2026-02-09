from pydantic import BaseModel


class RegisterRequest(BaseModel):
    learner_id: str
    tasks: list[str] | None = None
    outcomes: list[float] | None = None


class RegisterResponse(BaseModel):
    learner_id: str
    history_len: int


class PredictRequest(BaseModel):
    learner_id: str
    tasks: list[str]


class PredictResponse(BaseModel):
    learner_id: str
    predictions: list[float]


class UpdateRequest(BaseModel):
    learner_id: str
    task: str
    outcome: float


class UpdateResponse(BaseModel):
    learner_id: str
    history_len: int


class ForecastRequest(BaseModel):
    learner_id: str
    task_sequences: list[list[str]]


class ForecastResponse(BaseModel):
    learner_id: str
    predictions: list[list[float]]


class DeleteResponse(BaseModel):
    learner_id: str


class HealthResponse(BaseModel):
    status: str
    learner_count: int
