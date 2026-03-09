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


class AttentionRequest(BaseModel):
    learner_id: str
    target_task: str | None = None  # if provided, append as prediction query


class AttentionResponse(BaseModel):
    learner_id: str
    n_layers: int
    n_heads: int
    seq_len: int
    token_types: list[int]  # per-position: 0=BOS, 1=TASK, 2=OUTCOME, 3=ANSWER
    token_labels: list[str]  # human-readable label per position
    # fraction of total attention received by each token type, per layer
    attn_by_type: dict[str, list[float]]  # {type_name: [fraction_per_layer]}
    # same but only from the last position (the prediction position)
    last_pos_attn_by_type: dict[str, list[float]]  # {type_name: [fraction_per_layer]}
    # per-position attention from last position, averaged across layers and heads
    last_pos_attn_per_position: list[float]  # [weight_per_position]
    # mean attention distance (in positions) per layer — how far back each position looks
    mean_attn_distance: list[float]  # [mean_distance_per_layer]
    last_pos_attn_distance: list[float]  # same but from last position only


class HealthResponse(BaseModel):
    status: str
    learner_count: int
