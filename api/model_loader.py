from pathlib import Path
from typing import Any

_MODELS_DIR = Path(__file__).parent.parent / "models"
_registry: dict[str, Any] = {}


def _load(key: str, path: Path):
    if key in _registry:
        return _registry[key]

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.eval()

    _registry[key] = (tokenizer, model)
    return _registry[key]


def get_task1_encoder():
    path = _MODELS_DIR / "task1_encoder"
    if not path.exists():
        raise FileNotFoundError(f"task1_encoder not found at {path}")
    return _load("task1_encoder", path)


def get_task2_encoder():
    path = _MODELS_DIR / "task2_encoder"
    if not path.exists():
        raise FileNotFoundError(f"task2_encoder not found at {path}")
    return _load("task2_encoder", path)
