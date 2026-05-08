"""Mock inference layer for the thesis deployment dashboard.

The functions in this module intentionally mimic the contract expected from a
future production backend. Today they use deterministic heuristics so the
Streamlit demo can be reviewed without model artifacts, API keys, or GPU access.
Later, each function can delegate to the real Task 1 and Task 2 pipelines while
keeping the same response shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import re
from typing import Iterable


RHETORICAL_LABELS = {
    "INTRO": {"name": "Introduccion", "color": "#2563eb"},
    "BACK": {"name": "Antecedentes", "color": "#7c3aed"},
    "METH": {"name": "Metodologia", "color": "#0891b2"},
    "RES": {"name": "Resultados", "color": "#16a34a"},
    "DISC": {"name": "Discusion", "color": "#ca8a04"},
    "CONTR": {"name": "Contribucion", "color": "#dc2626"},
    "LIM": {"name": "Limitaciones", "color": "#9333ea"},
    "CONC": {"name": "Conclusiones", "color": "#475569"},
}


@dataclass(frozen=True)
class DemoModel:
    """Metadata shown in the dashboard and used for mock variation."""

    id: str
    name: str
    category: str
    task: str
    description: str
    latency_ms: int
    relative_cost: str


TASK1_MODELS = [
    DemoModel(
        "encoder-scibeto-task1",
        "SciBETO Task 1",
        "Encoder ajustado",
        "task1",
        "Clasificador multicategoria para etiquetas retoricas.",
        180,
        "Bajo",
    ),
    DemoModel(
        "openweight-llama-task1",
        "Open-weight 8B Task 1",
        "Decoder open-weight",
        "task1",
        "Modelo generativo pequeno con prompt de clasificacion.",
        860,
        "Medio",
    ),
    DemoModel(
        "commercial-api-task1",
        "API comercial Task 1",
        "LLM comercial via API",
        "task1",
        "Modelo remoto para clasificacion con explicacion corta.",
        1250,
        "Variable",
    ),
]

TASK2_MODELS = [
    DemoModel(
        "encoder-beto-task2",
        "BETO/SciBETO Task 2",
        "Encoder ajustado",
        "task2",
        "Clasificador binario contribucion/no contribucion.",
        120,
        "Bajo",
    ),
    DemoModel(
        "openweight-llama-task2",
        "Open-weight 8B Task 2",
        "Decoder open-weight",
        "task2",
        "Inferencia generativa con salida binaria y confianza.",
        780,
        "Medio",
    ),
    DemoModel(
        "commercial-api-task2",
        "API comercial Task 2",
        "LLM comercial via API",
        "task2",
        "Evaluacion remota de contribuciones explicitas.",
        1100,
        "Variable",
    ),
]


_LABEL_KEYWORDS = {
    "INTRO": ["introduccion", "objetivo", "problema", "motivacion", "necesidad", "este trabajo aborda"],
    "BACK": ["estado del arte", "literatura", "trabajos previos", "antecedentes", "se ha estudiado"],
    "METH": ["metodologia", "metodo", "dataset", "corpus", "entrenamiento", "modelo", "experimento"],
    "RES": ["resultado", "precision", "recall", "f1", "evidencia", "obtuvo", "alcanzo", "mejora"],
    "DISC": ["discusion", "interpretacion", "implica", "comparacion", "sugiere", "analisis"],
    "CONTR": ["contribucion", "aporte", "proponemos", "se propone", "desarrollamos", "presentamos"],
    "LIM": ["limitacion", "amenaza", "trabajo futuro", "restriccion", "no fue posible"],
    "CONC": ["conclusion", "en conclusion", "finalmente", "se concluye", "en sintesis"],
}

_CONTRIBUTION_KEYWORDS = [
    "contribucion",
    "aporte",
    "aportamos",
    "proponemos",
    "se propone",
    "desarrollamos",
    "presentamos",
    "nuevo",
    "novedoso",
    "marco",
    "dataset",
    "corpus anotado",
    "modelo",
    "metodo",
    "herramienta",
    "demostrador",
]


def get_model_options(task: str) -> list[DemoModel]:
    """Return available mock models for one task."""

    if task == "task1":
        return TASK1_MODELS
    if task == "task2":
        return TASK2_MODELS
    raise ValueError(f"Unsupported task: {task}")


def analyze_document(text: str, task1_model_id: str, task2_model_id: str) -> dict:
    """Run the mock Task 1 -> Task 2 pipeline over an academic text."""

    segments = []
    for index, paragraph in enumerate(split_into_paragraphs(text), start=1):
        task1_prediction = predict_rhetorical_label(paragraph, task1_model_id)
        task2_prediction = predict_contribution(paragraph, task2_model_id, task1_prediction["label"])
        segments.append(
            {
                "id": f"p{index}",
                "position": index,
                "text": paragraph,
                "task1": task1_prediction,
                "task2": task2_prediction,
            }
        )

    return {
        "input": {"paragraph_count": len(segments), "character_count": len(text)},
        "models": {
            "task1": model_to_dict(find_model(task1_model_id, TASK1_MODELS)),
            "task2": model_to_dict(find_model(task2_model_id, TASK2_MODELS)),
        },
        "segments": segments,
        "summary": build_summary(segments),
    }


def compare_task2_models(text: str, task1_model_id: str) -> list[dict]:
    """Compare all Task 2 model categories over the same Task 1 segmentation."""

    rows = []
    for index, paragraph in enumerate(split_into_paragraphs(text), start=1):
        task1_prediction = predict_rhetorical_label(paragraph, task1_model_id)
        row = {
            "Parrafo": f"P{index}",
            "Etiqueta retorica": task1_prediction["label"],
            "Texto": compact_text(paragraph, 110),
        }
        for model in TASK2_MODELS:
            prediction = predict_contribution(paragraph, model.id, task1_prediction["label"])
            verdict = "Si" if prediction["is_contribution"] else "No"
            row[model.name] = f"{verdict} ({prediction['confidence']:.2f})"
        rows.append(row)
    return rows


def split_into_paragraphs(text: str) -> list[str]:
    """Split user text into stable units for the mock pipeline."""

    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if not sentence:
            continue
        current.append(sentence)
        if len(" ".join(current).split()) >= 70:
            chunks.append(" ".join(current).strip())
            current = []
    if current:
        chunks.append(" ".join(current).strip())
    return chunks or [normalized]


def predict_rhetorical_label(paragraph: str, model_id: str) -> dict:
    """Produce a deterministic rhetorical label with model-specific variation."""

    text = normalize(paragraph)
    scores = {label: keyword_score(text, keywords) for label, keywords in _LABEL_KEYWORDS.items()}
    if max(scores.values()) == 0:
        scores["INTRO"] = 0.45
        scores["DISC"] = 0.35

    label = max(scores, key=scores.get)
    confidence = min(0.96, 0.54 + scores[label] * 0.12 + stable_noise(paragraph, model_id, 0.11))
    if model_id == "openweight-llama-task1" and label in {"RES", "DISC"}:
        confidence = max(0.51, confidence - 0.08)
    if model_id == "commercial-api-task1":
        confidence = min(0.98, confidence + 0.06)

    return {
        "label": label,
        "label_name": RHETORICAL_LABELS[label]["name"],
        "confidence": round(confidence, 3),
        "explanation": explain_rhetorical_label(label),
    }


def predict_contribution(paragraph: str, model_id: str, rhetorical_label: str) -> dict:
    """Produce a deterministic binary contribution prediction."""

    text = normalize(paragraph)
    score = keyword_score(text, _CONTRIBUTION_KEYWORDS)
    if rhetorical_label == "CONTR":
        score += 2.2
    if rhetorical_label in {"METH", "RES", "DISC"}:
        score += 0.55
    if rhetorical_label in {"BACK", "INTRO", "CONC"}:
        score -= 0.25

    threshold = 1.35
    if model_id == "openweight-llama-task2":
        threshold = 1.55
    if model_id == "commercial-api-task2":
        threshold = 1.15

    is_contribution = score >= threshold
    confidence = 0.5 + min(abs(score - threshold) * 0.14, 0.35)
    confidence += stable_noise(paragraph, model_id, 0.08)
    if model_id == "commercial-api-task2":
        confidence += 0.04
    if model_id == "openweight-llama-task2":
        confidence -= 0.03
    confidence = max(0.5, min(0.98, confidence))

    return {
        "is_contribution": is_contribution,
        "label": "Contribucion explicita" if is_contribution else "No contribucion",
        "confidence": round(confidence, 3),
        "evidence": explain_contribution(is_contribution, rhetorical_label, score),
    }


def build_summary(segments: Iterable[dict]) -> dict:
    """Aggregate segment-level predictions for dashboard metrics."""

    segment_list = list(segments)
    label_counts: dict[str, int] = {}
    contribution_count = 0
    confidence_values = []
    for segment in segment_list:
        label = segment["task1"]["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        contribution_count += int(segment["task2"]["is_contribution"])
        confidence_values.append(segment["task2"]["confidence"])

    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
    return {
        "segments": len(segment_list),
        "rhetorical_distribution": label_counts,
        "contribution_segments": contribution_count,
        "avg_task2_confidence": round(avg_confidence, 3),
    }


def find_model(model_id: str, models: Iterable[DemoModel]) -> DemoModel:
    """Return model metadata by id."""

    for model in models:
        if model.id == model_id:
            return model
    raise ValueError(f"Unknown model id: {model_id}")


def model_to_dict(model: DemoModel) -> dict:
    """Serialize model metadata for display or JSON export."""

    return {
        "id": model.id,
        "name": model.name,
        "category": model.category,
        "task": model.task,
        "description": model.description,
        "latency_ms": model.latency_ms,
        "relative_cost": model.relative_cost,
    }


def keyword_score(text: str, keywords: Iterable[str]) -> float:
    """Score text by keyword presence with a small repetition bonus."""

    score = 0.0
    for keyword in keywords:
        occurrences = text.count(keyword)
        if occurrences:
            score += 1.0 + min(occurrences - 1, 2) * 0.3
    return score


def normalize(text: str) -> str:
    """Normalize Spanish text for lightweight matching."""

    replacements = str.maketrans({"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"})
    return text.lower().translate(replacements)


def stable_noise(text: str, model_id: str, scale: float) -> float:
    """Small deterministic variation so mock models do not look identical."""

    digest = sha1(f"{model_id}:{text}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return (value - 0.5) * scale


def explain_rhetorical_label(label: str) -> str:
    """Short explanation aligned with the proposal's qualitative analysis goal."""

    explanations = {
        "INTRO": "El fragmento enmarca el problema u objetivo de investigacion.",
        "BACK": "El fragmento referencia contexto, literatura o trabajos relacionados.",
        "METH": "El fragmento describe datos, modelos, procedimiento o diseno experimental.",
        "RES": "El fragmento reporta hallazgos, metricas o evidencia empirica.",
        "DISC": "El fragmento interpreta resultados o compara implicaciones.",
        "CONTR": "El fragmento formula un aporte original del trabajo.",
        "LIM": "El fragmento expresa restricciones, amenazas o trabajo futuro.",
        "CONC": "El fragmento sintetiza el cierre o conclusiones del estudio.",
    }
    return explanations[label]


def explain_contribution(is_contribution: bool, rhetorical_label: str, score: float) -> str:
    """Short mock rationale for Task 2 predictions."""

    if is_contribution:
        if rhetorical_label == "CONTR":
            return "Coinciden senales de aporte explicito y etiqueta retorica de contribucion."
        return "El parrafo contiene senales asociadas a metodos, recursos o resultados originales."
    if score > 0.8:
        return "Hay senales parciales, pero no suficientes para marcar aporte explicito."
    return "El contenido parece contextual, descriptivo o conclusivo sin aporte explicito."


def compact_text(text: str, max_chars: int) -> str:
    """Trim long text for tables without losing paragraph identity."""

    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."
