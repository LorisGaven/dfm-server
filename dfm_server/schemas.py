from pydantic import BaseModel


class RegisterRequest(BaseModel):
    learner_id: str
    tasks: list[str] | None = None
    outcomes: list[float] | None = None
    answers: list[str] | None = None


class RegisterResponse(BaseModel):
    learner_id: str
    history_len: int


class PredictRequest(BaseModel):
    learner_id: str
    curriculum: list[list[str]] | None = None
    target_tasks: list[str] | None = None


class PredictResponse(BaseModel):
    learner_id: str
    predictions: list[list[float]]


class UpdateRequest(BaseModel):
    learner_id: str
    tasks: list[str]
    outcomes: list[float]
    answers: list[str] | None = None


class UpdateResponse(BaseModel):
    learner_id: str
    history_len: int


class DeleteResponse(BaseModel):
    learner_id: str


class HealthResponse(BaseModel):
    status: str
    learner_count: int
