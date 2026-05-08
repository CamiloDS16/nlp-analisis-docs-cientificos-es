import json
import os
import re
from collections import Counter
from pathlib import Path

from api.gemini_config import (
    GEMINI_MODEL,
    T1_MAX_WORDS,
    T1_MAX_TOKENS,
    T1_SYSTEM,
    T1_TEMPERATURE,
    T1_VOTING_K,
    T2_MAX_TOKENS,
    T2_MAX_WORDS,
    T2_SYSTEM,
    T2_TEMPERATURE,
)
from api.schemas import Task1Prediction, Task2Prediction

_T1_LABELS = ["INTRO", "BACK", "METH", "RES", "DISC", "CONTR", "LIM", "CONC"]

_T1_LABEL_NAMES = {
    "INTRO": "Introduccion",
    "BACK":  "Antecedentes",
    "METH":  "Metodologia",
    "RES":   "Resultados",
    "DISC":  "Discusion",
    "CONTR": "Contribucion",
    "LIM":   "Limitaciones",
    "CONC":  "Conclusiones",
}

_FEWSHOT_EXAMPLES: dict[str, str] = json.loads(
    (Path(__file__).parent / "fewshot_examples.json").read_text(encoding="utf-8")
)


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def _parse_json(raw: str) -> dict | None:
    raw = re.sub(r"```(?:json)?\n?", "", raw).replace("```", "").strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _gemini_client():
    from google import genai
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise EnvironmentError("GOOGLE_API_KEY not set")
    return genai.Client(api_key=key)


def _build_fewshot_contents(text: str) -> list:
    from google.genai import types

    def user_msg(t: str) -> types.Content:
        words = t.split()
        t = " ".join(words[:T1_MAX_WORDS]) if len(words) > T1_MAX_WORDS else t
        return types.Content(
            role="user",
            parts=[types.Part(text=f"Clasifica el siguiente fragmento de un artículo científico en español:\n\n{t}")]
        )

    contents = []
    for label in _T1_LABELS:
        example = _FEWSHOT_EXAMPLES.get(label, "")
        if example:
            contents.append(user_msg(example))
            contents.append(types.Content(
                role="model",
                parts=[types.Part(text=json.dumps({"label": label, "confidence": 1.0}))]
            ))
    contents.append(user_msg(text))
    return contents


def _gemini_t1(text: str) -> Task1Prediction:
    from google.genai import types

    client = _gemini_client()
    cfg = types.GenerateContentConfig(
        system_instruction=T1_SYSTEM,
        temperature=T1_TEMPERATURE,
        max_output_tokens=T1_MAX_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    contents = _build_fewshot_contents(_truncate(text, T1_MAX_WORDS))

    votes = []
    for _ in range(T1_VOTING_K):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=cfg,
            )
            parsed = _parse_json(response.text.strip())
            if parsed:
                label = parsed.get("label", "").upper()
                if label in _T1_LABELS:
                    votes.append({"label": label, "confidence": float(parsed.get("confidence", 0.8))})
        except Exception:
            pass

    if not votes:
        return Task1Prediction(label="INTRO", label_name="Introduccion", confidence=0.0, explanation="all votes failed")

    label_counts = Counter(v["label"] for v in votes)
    winner_count = max(label_counts.values())
    candidates = [l for l, c in label_counts.items() if c == winner_count]
    winner_label = max(
        candidates,
        key=lambda l: sum(v["confidence"] for v in votes if v["label"] == l)
    )
    winner_votes = [v for v in votes if v["label"] == winner_label]
    avg_conf = sum(v["confidence"] for v in winner_votes) / len(winner_votes)

    return Task1Prediction(
        label=winner_label,
        label_name=_T1_LABEL_NAMES[winner_label],
        confidence=round(avg_conf, 4),
        explanation="",
    )


def _gemini_t2(text: str, rhetorical_label: str = "") -> Task2Prediction:
    from google.genai import types

    client = _gemini_client()
    cfg = types.GenerateContentConfig(
        system_instruction=T2_SYSTEM,
        temperature=T2_TEMPERATURE,
        max_output_tokens=T2_MAX_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    context = f"Sección retórica: {rhetorical_label}.\n\n" if rhetorical_label else ""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Determina si el siguiente fragmento es una contribución científica:\n\n{context}{_truncate(text, T2_MAX_WORDS)}",
        config=cfg,
    )
    parsed = _parse_json(response.text.strip())
    if parsed:
        label = int(parsed.get("label", 0))
        conf  = float(parsed.get("confidence", 0.7))
        return Task2Prediction(
            is_contribution=bool(label),
            label="Contribucion" if label else "No contribucion",
            confidence=conf,
            evidence="",
        )
    return Task2Prediction(is_contribution=False, label="No contribucion", confidence=0.0, evidence="parse error")


def _encoder_t1(text: str) -> Task1Prediction:
    from api.model_loader import get_task1_encoder
    import torch

    tokenizer, model = get_task1_encoder()
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
        ).logits
    final_prob = torch.softmax(logits, dim=-1).mean(dim=0)
    idx = int(final_prob.argmax())
    label = model.config.id2label[idx]
    return Task1Prediction(
        label=label,
        label_name=_T1_LABEL_NAMES.get(label, label),
        confidence=round(float(final_prob[idx]), 4),
        explanation="",
    )


def _encoder_t2(text: str, rhetorical_label: str = "") -> Task2Prediction:
    from api.model_loader import get_task2_encoder
    import torch

    tokenizer, model = get_task2_encoder()
    prompt = f"{rhetorical_label} {tokenizer.sep_token} {_truncate(text, T2_MAX_WORDS)}" if rhetorical_label else _truncate(text, T2_MAX_WORDS)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    is_contr = bool(probs[1] > 0.5)
    return Task2Prediction(
        is_contribution=is_contr,
        label="Contribucion" if is_contr else "No contribucion",
        confidence=round(float(probs[1]), 4),
        evidence="",
    )


def _ollama_t1(text: str) -> Task1Prediction:
    import httpx

    url   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL_T1", "")
    if not model:
        raise EnvironmentError("OLLAMA_MODEL_T1 not set")

    prompt = f"{T1_SYSTEM}\n\nFragmento:\n{_truncate(text, T1_MAX_WORDS)}\n\nJSON:"
    resp = httpx.post(
        f"{url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    raw    = resp.json().get("response", "")
    parsed = _parse_json(raw)
    if parsed:
        label = parsed.get("label", "").upper()
        if label in _T1_LABELS:
            return Task1Prediction(
                label=label,
                label_name=_T1_LABEL_NAMES[label],
                confidence=float(parsed.get("confidence", 0.7)),
                explanation="",
            )
    return Task1Prediction(label="INTRO", label_name="Introduccion", confidence=0.0, explanation="parse error")


def _ollama_t2(text: str, rhetorical_label: str = "") -> Task2Prediction:
    import httpx

    url   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL_T2", "")
    if not model:
        raise EnvironmentError("OLLAMA_MODEL_T2 not set")

    label_ctx = f"Sección retórica: {rhetorical_label}.\n\n" if rhetorical_label else ""
    prompt = f"{T2_SYSTEM}\n\n{label_ctx}Fragmento:\n{_truncate(text, T2_MAX_WORDS)}\n\nJSON:"
    resp = httpx.post(
        f"{url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    raw    = resp.json().get("response", "")
    parsed = _parse_json(raw)
    if parsed:
        label  = int(parsed.get("label", 0))
        conf   = float(parsed.get("confidence", 0.7))
        return Task2Prediction(
            is_contribution=bool(label),
            label="Contribucion" if label else "No contribucion",
            confidence=conf,
            evidence="",
        )
    return Task2Prediction(is_contribution=False, label="No contribucion", confidence=0.0, evidence="parse error")


_T1_DISPATCH = {
    "commercial-api-gemini-task1": _gemini_t1,
    "encoder-scibeto-task1":       _encoder_t1,
    "openweight-task1":            _ollama_t1,
}

_T2_DISPATCH = {
    "commercial-api-gemini-task2": _gemini_t2,
    "encoder-beto-task2":          _encoder_t2,
    "openweight-task2":            _ollama_t2,
}


def predict_t1(text: str, model_id: str) -> Task1Prediction:
    fn = _T1_DISPATCH.get(model_id)
    if fn is None:
        raise ValueError(f"unknown task1 model: {model_id}")
    return fn(text)


def predict_t2(text: str, model_id: str, rhetorical_label: str = "") -> Task2Prediction:
    fn = _T2_DISPATCH.get(model_id)
    if fn is None:
        raise ValueError(f"unknown task2 model: {model_id}")
    if model_id in ("commercial-api-gemini-task2", "openweight-task2", "encoder-beto-task2"):
        return fn(text, rhetorical_label)
    return fn(text)
