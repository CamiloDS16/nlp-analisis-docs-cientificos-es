import json
import os
import re
import time

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

_T1_LABEL_ES = {
    "INTRO": "Introducción",
    "BACK":  "Antecedentes / revisión de literatura",
    "METH":  "Metodología",
    "RES":   "Resultados",
    "DISC":  "Discusión / interpretación de resultados",
    "CONTR": "Contribuciones del trabajo",
    "LIM":   "Limitaciones / trabajo futuro",
    "CONC":  "Conclusiones",
}

_T1_SYSTEM = """Eres un experto en análisis del discurso científico en español.
Clasifica el fragmento según su función retórica en uno de estos ocho tipos:
INTRO, BACK, METH, RES, DISC, CONTR, LIM, CONC.

INTRO: presenta el problema, motivación u objetivos del trabajo.
BACK: describe trabajos previos de otros autores; revisión de literatura.
METH: explica el diseño experimental, métodos, datos o procedimientos.
RES: presenta resultados empíricos sin interpretación extensiva.
DISC: interpreta resultados, analiza implicaciones, compara con literatura.
CONTR: declara explícitamente los aportes originales del trabajo.
LIM: describe restricciones, supuestos o trabajo futuro.
CONC: resume hallazgos y presenta conclusiones finales.

Responde ÚNICAMENTE con JSON: {"label": "<ETIQUETA>", "confidence": <0.0-1.0>}
Sin explicaciones ni markdown."""

_T2_SYSTEM = """Eres un experto en análisis del discurso científico en español.
Determina si el fragmento declara explícitamente una CONTRIBUCIÓN CIENTÍFICA ORIGINAL.

label=1 si contiene expresiones como: "proponemos", "nuestra contribución es",
"presentamos un nuevo método", "el aporte principal de este trabajo",
"a diferencia de trabajos previos, nuestro enfoque".

label=0 si solo presenta resultados, discute implicaciones, describe metodología,
revisa literatura, resume conclusiones o habla de limitaciones.

Responde ÚNICAMENTE con JSON: {"label": <0 o 1>, "confidence": <0.0-1.0>}
Sin explicaciones ni markdown."""

_MAX_WORDS = 700


def _truncate(text: str) -> str:
    words = text.split()
    return " ".join(words[:_MAX_WORDS]) if len(words) > _MAX_WORDS else text


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


# ── Gemini ────────────────────────────────────────────────────────────────────

def _gemini_t1(text: str) -> Task1Prediction:
    from google.genai import types

    client = _gemini_client()
    cfg = types.GenerateContentConfig(
        system_instruction=_T1_SYSTEM,
        temperature=0,
        max_output_tokens=60,
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"Clasifica el siguiente fragmento:\n\n{_truncate(text)}",
        config=cfg,
    )
    parsed = _parse_json(response.text.strip())
    if parsed:
        label = parsed.get("label", "").upper()
        if label in _T1_LABELS:
            return Task1Prediction(
                label=label,
                label_name=_T1_LABEL_NAMES[label],
                confidence=float(parsed.get("confidence", 0.8)),
                explanation="",
            )
    return Task1Prediction(label="INTRO", label_name="Introduccion", confidence=0.0, explanation="parse error")


def _gemini_t2(text: str, rhetorical_label: str = "") -> Task2Prediction:
    from google.genai import types

    client = _gemini_client()
    cfg = types.GenerateContentConfig(
        system_instruction=_T2_SYSTEM,
        temperature=0,
        max_output_tokens=60,
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"Determina si el siguiente fragmento es una contribución científica:\n\n{_truncate(text)}",
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


# ── Encoder ───────────────────────────────────────────────────────────────────

def _encoder_t1(text: str) -> Task1Prediction:
    from api.model_loader import get_task1_encoder
    import torch

    tokenizer, model = get_task1_encoder()
    inputs = tokenizer(_truncate(text), return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs  = torch.softmax(logits, dim=-1)[0]
    idx    = int(probs.argmax())
    label  = model.config.id2label[idx]
    return Task1Prediction(
        label=label,
        label_name=_T1_LABEL_NAMES.get(label, label),
        confidence=round(float(probs[idx]), 4),
        explanation="",
    )


def _encoder_t2(text: str, rhetorical_label: str = "") -> Task2Prediction:
    from api.model_loader import get_task2_encoder
    import torch

    tokenizer, model = get_task2_encoder()
    prompt = f"{rhetorical_label} {tokenizer.sep_token} {_truncate(text)}" if rhetorical_label else _truncate(text)
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


# ── Open-weight via Ollama ────────────────────────────────────────────────────

def _ollama_t1(text: str) -> Task1Prediction:
    import httpx

    url   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL_T1", "")
    if not model:
        raise EnvironmentError("OLLAMA_MODEL_T1 not set")

    prompt = f"{_T1_SYSTEM}\n\nFragmento:\n{_truncate(text)}\n\nJSON:"
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
    prompt = f"{_T2_SYSTEM}\n\n{label_ctx}Fragmento:\n{_truncate(text)}\n\nJSON:"
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


# ── Dispatch ──────────────────────────────────────────────────────────────────

_T1_DISPATCH = {
    "commercial-api-gemini-task1":   _gemini_t1,
    "encoder-scibeto-task1":         _encoder_t1,
    "openweight-task1":              _ollama_t1,
}

_T2_DISPATCH = {
    "commercial-api-gemini-task2":   _gemini_t2,
    "encoder-beto-task2":            _encoder_t2,
    "openweight-task2":              _ollama_t2,
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
    # only gemini and ollama accept the rhetorical label context
    if model_id in ("commercial-api-gemini-task2", "openweight-task2", "encoder-beto-task2"):
        return fn(text, rhetorical_label)
    return fn(text)
