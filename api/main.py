import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.inference import predict_t1, predict_t2
from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CompareRequest,
    ModelInfo,
    Segment,
    Summary,
)

app = FastAPI(title="PLN Scientific Docs API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_RHETORICAL_LABELS = {
    "INTRO": {"name": "Introduccion",  "color": "#2563eb"},
    "BACK":  {"name": "Antecedentes",  "color": "#7c3aed"},
    "METH":  {"name": "Metodologia",   "color": "#0891b2"},
    "RES":   {"name": "Resultados",    "color": "#16a34a"},
    "DISC":  {"name": "Discusion",     "color": "#ca8a04"},
    "CONTR": {"name": "Contribucion",  "color": "#dc2626"},
    "LIM":   {"name": "Limitaciones",  "color": "#9333ea"},
    "CONC":  {"name": "Conclusiones",  "color": "#475569"},
}

_TASK1_MODELS = [
    ModelInfo(id="commercial-api-gemini-task1", name="Gemini Task 1",      category="LLM comercial via API",  description="Clasificacion retorica zero-shot con Gemini.",        latency_ms=900,  relative_cost="Variable"),
    ModelInfo(id="encoder-scibeto-task1",       name="SciBETO Task 1",     category="Encoder ajustado",       description="SciBETO fine-tuned para clasificacion retorica.",     latency_ms=180,  relative_cost="Bajo"),
    ModelInfo(id="openweight-task1",            name="Open-weight Task 1", category="Decoder open-weight",    description="LLM open-weight via Ollama.",                          latency_ms=860,  relative_cost="Bajo"),
]

_TASK2_MODELS = [
    ModelInfo(id="commercial-api-gemini-task2", name="Gemini Task 2",      category="LLM comercial via API",  description="Deteccion binaria CONTR zero-shot con Gemini.",       latency_ms=800,  relative_cost="Variable"),
    ModelInfo(id="encoder-beto-task2",          name="BETO/SciBETO Task 2",category="Encoder ajustado",       description="BETO fine-tuned para clasificacion binaria CONTR.",   latency_ms=120,  relative_cost="Bajo"),
    ModelInfo(id="openweight-task2",            name="Open-weight Task 2", category="Decoder open-weight",    description="LLM open-weight via Ollama con contexto retorico.",   latency_ms=780,  relative_cost="Bajo"),
]


def _split_paragraphs(text: str) -> list[str]:
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    sentences  = re.split(r"(?<=[.!?])\s+", normalized)
    chunk_size = max(2, len(sentences) // 5)
    return [
        " ".join(sentences[i: i + chunk_size])
        for i in range(0, len(sentences), chunk_size)
        if sentences[i: i + chunk_size]
    ]


def _build_summary(segments: list[Segment]) -> Summary:
    label_counts: dict[str, int] = {}
    contribution_count = 0
    total_t2_conf = 0.0
    for s in segments:
        lbl = s.task1.label
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        if s.task2.is_contribution:
            contribution_count += 1
        total_t2_conf += s.task2.confidence
    dominant = max(label_counts, key=label_counts.get) if label_counts else "INTRO"
    avg_conf = total_t2_conf / len(segments) if segments else 0.0
    return Summary(
        dominant_label=dominant,
        dominant_label_name=_RHETORICAL_LABELS.get(dominant, {}).get("name", dominant),
        segments=len(segments),
        contribution_segments=contribution_count,
        avg_task2_confidence=round(avg_conf, 4),
        rhetorical_distribution=label_counts,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models/{task}")
def list_models(task: str):
    if task == "task1":
        return [m.model_dump() for m in _TASK1_MODELS]
    if task == "task2":
        return [m.model_dump() for m in _TASK2_MODELS]
    raise HTTPException(status_code=404, detail=f"unknown task: {task}")


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    paragraphs = _split_paragraphs(req.text)
    if not paragraphs:
        raise HTTPException(status_code=422, detail="empty text")

    segments = []
    for i, para in enumerate(paragraphs, start=1):
        t1 = predict_t1(para, req.task1_model_id)
        t2 = predict_t2(para, req.task2_model_id, t1.label)
        segments.append(Segment(id=f"p{i}", position=i, text=para, task1=t1, task2=t2))

    t1_model = next((m for m in _TASK1_MODELS if m.id == req.task1_model_id), _TASK1_MODELS[0])
    t2_model = next((m for m in _TASK2_MODELS if m.id == req.task2_model_id), _TASK2_MODELS[0])

    return AnalyzeResponse(
        input={"paragraph_count": len(segments), "character_count": len(req.text)},
        models={"task1": t1_model, "task2": t2_model},
        segments=segments,
        summary=_build_summary(segments),
    )


@app.post("/compare")
def compare(req: CompareRequest):
    paragraphs = _split_paragraphs(req.text)
    if not paragraphs:
        raise HTTPException(status_code=422, detail="empty text")

    rows = []
    for i, para in enumerate(paragraphs, start=1):
        t1  = predict_t1(para, req.task1_model_id)
        row = {
            "Parrafo": f"P{i}",
            "Etiqueta retorica": t1.label,
            "Texto": para[:110] + "..." if len(para) > 110 else para,
        }
        for m in _TASK2_MODELS:
            t2      = predict_t2(para, m.id, t1.label)
            verdict = "Si" if t2.is_contribution else "No"
            row[m.name] = f"{verdict} ({t2.confidence:.2f})"
        rows.append(row)

    return rows
