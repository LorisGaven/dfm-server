"""End-to-end tests for DFM inference server using a synthetic model."""

import torch
from fastapi.testclient import TestClient

from dfm_server.model import DFM, DFMConfig, KVCache
from dfm_server import server as server_module
from dfm_server.server import app

# ---------------------------------------------------------------------------
# Synthetic setup: tiny model + fake embeddings
# ---------------------------------------------------------------------------

N_INPUT = 16
N_EMBD = 32
N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 2
BLOCK_SIZE = 64
N_TASKS = 10


def setup_synthetic():
    """Create a tiny DFM model and fake embedding table, inject into server module."""
    config = DFMConfig(
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_kv_head=N_KV_HEAD,
        n_embd=N_EMBD,
        n_input=N_INPUT,
    )
    model = DFM(config)
    model.eval()
    model.to(dtype=torch.bfloat16)

    # Create fake embeddings: row 0 = BOS zeros, rows 1..N_TASKS = random
    bos_row = torch.zeros(1, N_INPUT, dtype=torch.bfloat16)
    real_embs = torch.randn(N_TASKS, N_INPUT, dtype=torch.bfloat16)
    emb_tensor = torch.cat([bos_row, real_embs], dim=0)

    # Build task_to_idx: "task_0" -> 1, "task_1" -> 2, ...
    task_to_idx = {f"task_{i}": i + 1 for i in range(N_TASKS)}

    # Inject into server module
    server_module.model = model
    server_module.emb_tensor = emb_tensor
    server_module.task_to_idx = task_to_idx
    server_module.learners = {}
    server_module.MAX_SEQ_LEN = 256


setup_synthetic()
client = TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["learner_count"] == 0


def test_register_empty():
    r = client.post("/learners", json={"learner_id": "empty1"})
    assert r.status_code == 200
    data = r.json()
    assert data["learner_id"] == "empty1"
    assert data["history_len"] == 0
    # Cleanup
    client.delete("/learners/empty1")


def test_register_with_history():
    r = client.post("/learners", json={
        "learner_id": "hist1",
        "tasks": ["task_0", "task_1", "task_2"],
        "outcomes": [1.0, 0.0, 1.0],
    })
    assert r.status_code == 200
    data = r.json()
    assert data["history_len"] == 3
    client.delete("/learners/hist1")


def test_register_duplicate():
    client.post("/learners", json={"learner_id": "dup1"})
    r = client.post("/learners", json={"learner_id": "dup1"})
    assert r.status_code == 409
    client.delete("/learners/dup1")


def test_predict_returns_probabilities():
    client.post("/learners", json={"learner_id": "pred1"})
    r = client.post("/predict", json={
        "learner_id": "pred1",
        "tasks": ["task_0", "task_1", "task_2"],
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["predictions"]) == 3
    for p in data["predictions"]:
        assert 0.0 <= p <= 1.0, f"Prediction {p} out of range"
    client.delete("/learners/pred1")


def test_predict_is_non_destructive():
    """Calling predict twice should give identical results (cache not advanced)."""
    client.post("/learners", json={"learner_id": "nd1"})
    r1 = client.post("/predict", json={"learner_id": "nd1", "tasks": ["task_0"]})
    r2 = client.post("/predict", json={"learner_id": "nd1", "tasks": ["task_0"]})
    assert r1.json()["predictions"] == r2.json()["predictions"]
    client.delete("/learners/nd1")


def test_update_advances_history():
    client.post("/learners", json={"learner_id": "upd1"})
    r = client.post("/update", json={"learner_id": "upd1", "task": "task_0", "outcome": 1.0})
    assert r.status_code == 200
    assert r.json()["history_len"] == 1

    r = client.post("/update", json={"learner_id": "upd1", "task": "task_1", "outcome": 0.0})
    assert r.json()["history_len"] == 2
    client.delete("/learners/upd1")


def test_forecast_returns_correct_shape():
    client.post("/learners", json={"learner_id": "fc1"})
    r = client.post("/forecast", json={
        "learner_id": "fc1",
        "task_sequences": [
            ["task_0", "task_1", "task_2"],
            ["task_3", "task_4", "task_5"],
        ],
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["predictions"]) == 2  # S=2 sequences
    for seq_preds in data["predictions"]:
        assert len(seq_preds) == 3  # L=3 steps each
        for p in seq_preds:
            assert 0.0 <= p <= 1.0
    client.delete("/learners/fc1")


def test_forecast_is_non_destructive():
    """Forecast should not change the learner's cache."""
    client.post("/learners", json={"learner_id": "fcnd1"})
    client.post("/update", json={"learner_id": "fcnd1", "task": "task_0", "outcome": 1.0})

    # Predict before forecast
    r1 = client.post("/predict", json={"learner_id": "fcnd1", "tasks": ["task_1"]})
    # Forecast (should not mutate state)
    client.post("/forecast", json={
        "learner_id": "fcnd1",
        "task_sequences": [["task_1", "task_2"], ["task_3", "task_4"]],
    })
    # Predict after forecast — should match
    r2 = client.post("/predict", json={"learner_id": "fcnd1", "tasks": ["task_1"]})
    assert r1.json()["predictions"] == r2.json()["predictions"]
    client.delete("/learners/fcnd1")


def test_forecast_single_sequence_matches_sequential():
    """A single-sequence batched forecast should match doing it step-by-step manually."""
    tasks = ["task_0", "task_1"]
    outcomes = [1.0, 0.0]

    # Register two identical learners
    client.post("/learners", json={"learner_id": "fseq_a", "tasks": tasks, "outcomes": outcomes})
    client.post("/learners", json={"learner_id": "fseq_b", "tasks": tasks, "outcomes": outcomes})

    forecast_tasks = ["task_3", "task_4", "task_5"]

    # Path A: batched forecast with S=1
    r_a = client.post("/forecast", json={
        "learner_id": "fseq_a",
        "task_sequences": [forecast_tasks],
    })
    preds_a = r_a.json()["predictions"][0]

    # Path B: manual autoregressive loop (predict + update with predicted outcome)
    preds_b = []
    for t in forecast_tasks:
        r = client.post("/predict", json={"learner_id": "fseq_b", "tasks": [t]})
        p = r.json()["predictions"][0]
        preds_b.append(p)
        client.post("/update", json={"learner_id": "fseq_b", "task": t, "outcome": p})

    for i, (a, b) in enumerate(zip(preds_a, preds_b)):
        assert abs(a - b) < 1e-3, f"Step {i}: forecast={a}, sequential={b}, diff={abs(a-b)}"

    client.delete("/learners/fseq_a")
    client.delete("/learners/fseq_b")


def test_forecast_different_sequences_differ():
    """Different task sequences should (generally) produce different predictions."""
    client.post("/learners", json={"learner_id": "fdiff1"})
    r = client.post("/forecast", json={
        "learner_id": "fdiff1",
        "task_sequences": [
            ["task_0", "task_1"],
            ["task_5", "task_6"],
        ],
    })
    preds = r.json()["predictions"]
    # With random weights, different inputs should produce different outputs
    assert preds[0] != preds[1], "Different sequences produced identical predictions"
    client.delete("/learners/fdiff1")


def test_forecast_empty_sequences():
    client.post("/learners", json={"learner_id": "fempty1"})
    r = client.post("/forecast", json={
        "learner_id": "fempty1",
        "task_sequences": [[], []],
    })
    assert r.status_code == 200
    assert r.json()["predictions"] == [[], []]
    client.delete("/learners/fempty1")


def test_forecast_mismatched_lengths():
    client.post("/learners", json={"learner_id": "fmis1"})
    r = client.post("/forecast", json={
        "learner_id": "fmis1",
        "task_sequences": [["task_0", "task_1"], ["task_2"]],
    })
    assert r.status_code == 400
    client.delete("/learners/fmis1")


def test_delete_learner():
    client.post("/learners", json={"learner_id": "del1"})
    r = client.delete("/learners/del1")
    assert r.status_code == 200
    assert r.json()["learner_id"] == "del1"
    # Should be gone
    r = client.post("/predict", json={"learner_id": "del1", "tasks": ["task_0"]})
    assert r.status_code == 404


def test_delete_nonexistent():
    r = client.delete("/learners/nope")
    assert r.status_code == 404


def test_unknown_task():
    client.post("/learners", json={"learner_id": "unk1"})
    r = client.post("/predict", json={"learner_id": "unk1", "tasks": ["nonexistent_task"]})
    assert r.status_code == 400
    client.delete("/learners/unk1")


def test_consistency_prefill_vs_incremental():
    """Prefill-then-predict should match register-empty-then-update-one-by-one-then-predict.

    This is the key correctness test: the KV cache built incrementally via
    update() must produce the same predictions as the KV cache built via
    bulk prefill at registration time.
    """
    tasks = ["task_0", "task_1", "task_2"]
    outcomes = [1.0, 0.0, 1.0]
    predict_tasks = ["task_3", "task_4"]

    # Path A: register with full history, then predict
    client.post("/learners", json={
        "learner_id": "cons_prefill",
        "tasks": tasks,
        "outcomes": outcomes,
    })
    r_a = client.post("/predict", json={"learner_id": "cons_prefill", "tasks": predict_tasks})
    preds_a = r_a.json()["predictions"]

    # Path B: register empty, update one by one, then predict
    client.post("/learners", json={"learner_id": "cons_incr"})
    for t, o in zip(tasks, outcomes):
        client.post("/update", json={"learner_id": "cons_incr", "task": t, "outcome": o})
    r_b = client.post("/predict", json={"learner_id": "cons_incr", "tasks": predict_tasks})
    preds_b = r_b.json()["predictions"]

    # They should be very close (bf16 rounding may cause tiny diffs)
    for i, (a, b) in enumerate(zip(preds_a, preds_b)):
        assert abs(a - b) < 1e-3, f"Prediction {i}: prefill={a}, incremental={b}, diff={abs(a-b)}"

    client.delete("/learners/cons_prefill")
    client.delete("/learners/cons_incr")


def test_learner_count():
    """Health endpoint should reflect correct learner count."""
    assert client.get("/health").json()["learner_count"] == 0
    client.post("/learners", json={"learner_id": "cnt1"})
    client.post("/learners", json={"learner_id": "cnt2"})
    assert client.get("/health").json()["learner_count"] == 2
    client.delete("/learners/cnt1")
    assert client.get("/health").json()["learner_count"] == 1
    client.delete("/learners/cnt2")
    assert client.get("/health").json()["learner_count"] == 0


if __name__ == "__main__":
    import sys

    test_fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in test_fns:
        # Reset learners between tests
        server_module.learners = {}
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    sys.exit(1 if failed else 0)
