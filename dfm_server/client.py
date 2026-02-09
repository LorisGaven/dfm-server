"""Python client for the DFM inference server."""

import requests


class DFMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.url = base_url.rstrip("/")

    def health(self) -> dict:
        r = requests.get(f"{self.url}/health")
        r.raise_for_status()
        return r.json()

    def tasks(self) -> list[str]:
        r = requests.get(f"{self.url}/tasks")
        r.raise_for_status()
        return r.json()["tasks"]

    def config(self) -> dict:
        r = requests.get(f"{self.url}/config")
        r.raise_for_status()
        return r.json()

    def gc(self):
        r = requests.post(f"{self.url}/gc")
        r.raise_for_status()
        return r.json()

    def register(self, learner_id: str, tasks: list[str] | None = None, outcomes: list[float] | None = None) -> dict:
        body = {"learner_id": learner_id}
        if tasks is not None:
            body["tasks"] = tasks
            body["outcomes"] = outcomes
        r = requests.post(f"{self.url}/learners", json=body)
        r.raise_for_status()
        return r.json()

    def predict(self, learner_id: str, tasks: list[str]) -> list[float]:
        r = requests.post(f"{self.url}/predict", json={"learner_id": learner_id, "tasks": tasks})
        r.raise_for_status()
        return r.json()["predictions"]

    def update(self, learner_id: str, task: str, outcome: float) -> dict:
        r = requests.post(f"{self.url}/update", json={"learner_id": learner_id, "task": task, "outcome": outcome})
        r.raise_for_status()
        return r.json()

    def forecast(self, learner_id: str, task_sequences: list[list[str]]) -> list[list[float]]:
        r = requests.post(f"{self.url}/forecast", json={"learner_id": learner_id, "task_sequences": task_sequences})
        r.raise_for_status()
        return r.json()["predictions"]

    def delete(self, learner_id: str) -> dict:
        r = requests.delete(f"{self.url}/learners/{learner_id}")
        r.raise_for_status()
        return r.json()
