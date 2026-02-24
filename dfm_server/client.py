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

    def register(
        self,
        learner_id: str,
        tasks: list[str] | None = None,
        outcomes: list[float] | None = None,
        answers: list[str] | None = None,
    ) -> dict:
        body = {"learner_id": learner_id}
        if tasks is not None:
            body["tasks"] = tasks
            body["outcomes"] = outcomes
        if answers is not None:
            body["answers"] = answers
        r = requests.post(f"{self.url}/learners", json=body)
        r.raise_for_status()
        return r.json()

    def predict(
        self,
        learner_id: str,
        curriculum: list[list[str]] | None = None,
        target_tasks: list[str] | None = None,
    ) -> list[list[float]]:
        body: dict = {"learner_id": learner_id}
        if curriculum is not None:
            body["curriculum"] = curriculum
        if target_tasks is not None:
            body["target_tasks"] = target_tasks
        r = requests.post(f"{self.url}/predict", json=body)
        r.raise_for_status()
        return r.json()["predictions"]

    def update(
        self,
        learner_id: str,
        tasks: list[str],
        outcomes: list[float],
        answers: list[str] | None = None,
    ) -> dict:
        body: dict = {
            "learner_id": learner_id,
            "tasks": tasks,
            "outcomes": outcomes,
        }
        if answers is not None:
            body["answers"] = answers
        r = requests.post(f"{self.url}/update", json=body)
        r.raise_for_status()
        return r.json()

    def delete(self, learner_id: str) -> dict:
        r = requests.delete(f"{self.url}/learners/{learner_id}")
        r.raise_for_status()
        return r.json()
