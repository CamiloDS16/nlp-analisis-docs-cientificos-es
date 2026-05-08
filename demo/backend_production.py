from __future__ import annotations

import os
from dataclasses import dataclass

import requests

_BASE = os.environ.get("BACKEND_URL", "http://localhost:8000")


RHETORICAL_LABELS = {
    "INTRO": {"name": "Introduccion", "color": "#2563eb"},
    "BACK":  {"name": "Antecedentes", "color": "#7c3aed"},
    "METH":  {"name": "Metodologia",  "color": "#0891b2"},
    "RES":   {"name": "Resultados",   "color": "#16a34a"},
    "DISC":  {"name": "Discusion",    "color": "#ca8a04"},
    "CONTR": {"name": "Contribucion", "color": "#dc2626"},
    "LIM":   {"name": "Limitaciones", "color": "#9333ea"},
    "CONC":  {"name": "Conclusiones", "color": "#475569"},
}


@dataclass(frozen=True)
class DemoModel:
    id: str
    name: str
    category: str
    task: str
    description: str
    latency_ms: int
    relative_cost: str


_TASK1_MODELS = [
    DemoModel("commercial-api-gemini-task1", "Gemini Task 1",      "LLM comercial via API", "task1", "Clasificacion retorica con Gemini.",            900,  "Variable"),
    DemoModel("encoder-scibeto-task1",       "SciBETO Task 1",     "Encoder ajustado",      "task1", "SciBETO fine-tuned para clasificacion retorica.", 180, "Bajo"),
    DemoModel("openweight-task1",            "Open-weight Task 1", "Decoder open-weight",   "task1", "LLM open-weight via Ollama.",                    860,  "Bajo"),
]

_TASK2_MODELS = [
    DemoModel("commercial-api-gemini-task2", "Gemini Task 2",       "LLM comercial via API", "task2", "Deteccion binaria CONTR con Gemini.",            800,  "Variable"),
    DemoModel("encoder-beto-task2",          "BETO/SciBETO Task 2", "Encoder ajustado",      "task2", "BETO fine-tuned para clasificacion binaria.",    120,  "Bajo"),
    DemoModel("openweight-task2",            "Open-weight Task 2",  "Decoder open-weight",   "task2", "LLM open-weight via Ollama.",                    780,  "Bajo"),
]


def get_model_options(task: str) -> list[DemoModel]:
    if task == "task1":
        return _TASK1_MODELS
    if task == "task2":
        return _TASK2_MODELS
    raise ValueError(f"unknown task: {task}")


def analyze_document(text: str, task1_model_id: str, task2_model_id: str) -> dict:
    resp = requests.post(
        f"{_BASE}/analyze",
        json={"text": text, "task1_model_id": task1_model_id, "task2_model_id": task2_model_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def compare_task2_models(text: str, task1_model_id: str) -> list[dict]:
    resp = requests.post(
        f"{_BASE}/compare",
        json={"text": text, "task1_model_id": task1_model_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()
