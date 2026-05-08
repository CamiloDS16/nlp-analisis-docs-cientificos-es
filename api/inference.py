import json
import os
import re
from pathlib import Path

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

_T1_SYSTEM = """Eres un experto en análisis del discurso científico en español.
Clasifica el fragmento textual de un artículo científico en una de estas 8 categorías retóricas.

DEFINICIONES Y SEÑALES PRIMARIAS:

CONTR — El fragmento DECLARA EXPLÍCITAMENTE el aporte original del trabajo.
  Señales: verbos en primera persona del plural referidos al propio trabajo ("proponemos",
  "presentamos", "desarrollamos", "describimos"), frases como "nuestra contribución es",
  "a diferencia de trabajos previos, este trabajo/método/sistema", "el aporte principal
  de este artículo es", "este trabajo introduce/propone/presenta un nuevo".
  NO es INTRO (que plantea objetivos sin declarar aportes) ni DISC (que interpreta resultados).
  REGLA: ante cualquier señal de declaración de aporte original, clasifica como CONTR.

BACK — Describe trabajos PREVIOS de OTROS autores. No habla del estudio actual.
  Señal principal: citas bibliográficas en cualquier formato ([1], (Autor, año), [Autor et al.],
  "según X", "Y et al. demostraron", "estudios previos de Z").
  Si el fragmento menciona trabajos ajenos o incluye citas, es BACK aunque parezca INTRO.
  NO es INTRO (sin citas, habla del problema actual) ni DISC (interpreta resultados propios).

METH — Explica qué se hizo: diseño experimental, métodos, datos, materiales, procedimientos.
  Describe los pasos seguidos para realizar el estudio.
  NO es LIM: si el énfasis está en restricciones o fallas del método, es LIM, no METH.

RES — Presenta solo los resultados obtenidos: números, porcentajes, tablas, comparaciones empíricas.
  Sin interpretación de qué significan esos datos.
  NO es DISC: si el fragmento dice "esto sugiere", "esto indica", "esto demuestra", es DISC.
  NO es BACK: si los resultados son del propio estudio (no cita trabajos ajenos), es RES.

DISC — Interpreta los resultados del estudio actual, analiza implicaciones, compara con hipótesis.
  Frases como "estos resultados sugieren", "esto indica que", "en comparación con [hipótesis]".
  NO es RES (que solo reporta datos sin interpretar).

LIM — Describe restricciones, supuestos, fuentes de error o limitaciones de generalización.
  Puede mencionar trabajo futuro. El énfasis está en qué NO funciona o qué es imperfecto.
  Frases como "una limitación de este estudio es", "no consideramos", "queda pendiente".
  NO es METH (que describe el método sin señalar sus restricciones).

CONC — Resume los hallazgos principales y presenta las conclusiones finales.
  Suele aparecer al final del artículo. Frases como "en conclusión", "este trabajo demostró".
  NO es DISC (que interpreta resultados) ni LIM (que solo menciona limitaciones).

INTRO — Presenta el problema de investigación, motivación y objetivos del trabajo.
  NO cita trabajos ajenos (eso es BACK). NO declara aportes propios (eso es CONTR).
  Usa INTRO solo si el fragmento plantea el problema o los objetivos sin señales de BACK ni CONTR.

ORDEN DE PRIORIDAD (aplica en orden ante la duda):
1. ¿Declara explícitamente un aporte original? → CONTR
2. ¿Cita trabajos de otros autores? → BACK
3. ¿Resume hallazgos finales o concluye el artículo? → CONC
4. ¿Presenta solo datos sin interpretar? → RES
5. ¿Describe métodos sin mencionar restricciones? → METH
6. ¿Interpreta qué significan los resultados? → DISC
7. ¿Menciona restricciones o limitaciones? → LIM
8. Ninguno de los anteriores → INTRO

Responde ÚNICAMENTE con un JSON válido en el formato exacto:
{"label": "<UNA DE LAS 8 ETIQUETAS>", "confidence": <número entre 0.0 y 1.0>}

No incluyas explicaciones, markdown ni texto fuera del JSON."""

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


def _build_fewshot_contents(text: str) -> list:
    from google.genai import types

    def user_msg(t: str) -> types.Content:
        words = t.split()
        t = " ".join(words[:700]) if len(words) > 700 else t
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
        system_instruction=_T1_SYSTEM,
        temperature=0,
        max_output_tokens=150,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=_build_fewshot_contents(_truncate(text)),
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
        max_output_tokens=150,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    context = f"Sección retórica: {rhetorical_label}.\n\n" if rhetorical_label else ""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Determina si el siguiente fragmento es una contribución científica:\n\n{context}{_truncate(text)}",
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
    if model_id in ("commercial-api-gemini-task2", "openweight-task2", "encoder-beto-task2"):
        return fn(text, rhetorical_label)
    return fn(text)
