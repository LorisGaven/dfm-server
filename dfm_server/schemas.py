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


class SearchRequest(BaseModel):
    learner_id: str
    target_tasks: list[str]
    candidate_tasks: list[str]
    depth: int
    population_size: int = 64
    generations: int = 20
    elite_count: int = 4
    tournament_size: int = 2
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    eval_every: int | None = None
    seed: int | None = None


class SearchResponse(BaseModel):
    learner_id: str
    best_sequence: list[str]
    best_fitness: float


class HealthResponse(BaseModel):
    status: str
    learner_count: int
