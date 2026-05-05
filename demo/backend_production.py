"""
Production replacement for mock_backend.py.
Proxies all calls to the FastAPI backend running at BACKEND_URL.

To activate:
    cp demo/backend_production.py demo/mock_backend.py
"""

import os

import requests

from demo.mock_backend import (  # keeps RHETORICAL_LABELS, DemoModel, get_model_options
    RHETORICAL_LABELS,
    DemoModel,
    get_model_options,
)

_BASE = os.environ.get("BACKEND_URL", "http://localhost:8000")

_DEFAULT_T1 = "commercial-api-gemini-task1"
_DEFAULT_T2 = "commercial-api-gemini-task2"


def analyze_document(text: str, task1_model_id: str, task2_model_id: str) -> dict:
    resp = requests.post(
        f"{_BASE}/analyze",
        json={
            "text": text,
            "task1_model_id": task1_model_id,
            "task2_model_id": task2_model_id,
        },
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
