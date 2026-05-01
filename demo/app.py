"""Streamlit dashboard for the thesis deployment mock."""

from __future__ import annotations

import html
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from mock_backend import RHETORICAL_LABELS, analyze_document, compare_task2_models, get_model_options  # noqa: E402
from sample_inputs import SAMPLE_TEXTS  # noqa: E402


st.set_page_config(
    page_title="Demo PLN Cientifico ES",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
    .app-title { font-size: 1.75rem; font-weight: 750; margin-bottom: 0.15rem; }
    .app-subtitle { color: #475569; font-size: 0.98rem; margin-bottom: 1rem; }
    .segment-card {
        border: 1px solid #d9e2ec;
        border-left: 6px solid var(--label-color);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
        background: #ffffff;
    }
    .segment-header { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin-bottom: 0.65rem; }
    .badge {
        border-radius: 999px;
        padding: 0.18rem 0.55rem;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid #cbd5e1;
        background: #f8fafc;
        color: #334155;
    }
    .label-badge { color: #ffffff; background: var(--label-color); border-color: var(--label-color); }
    .positive { color: #166534; background: #dcfce7; border-color: #86efac; }
    .negative { color: #991b1b; background: #fee2e2; border-color: #fecaca; }
    .segment-text { color: #0f172a; line-height: 1.55; margin-bottom: 0.65rem; }
    .explain { color: #475569; font-size: 0.9rem; line-height: 1.45; }
    .legend-row { display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.35rem; font-size: 0.9rem; }
    .legend-dot {
        width: 0.7rem;
        height: 0.7rem;
        border-radius: 999px;
        background: var(--label-color);
        display: inline-block;
    }
</style>
"""


def main() -> None:
    """Render the dashboard."""

    st.markdown(CSS, unsafe_allow_html=True)
    render_header()

    with st.sidebar:
        text, task1_model_id, task2_model_id = render_sidebar()

    if not text.strip():
        st.info("Ingresa un texto academico o selecciona un ejemplo para iniciar el analisis.")
        return

    result = analyze_document(text, task1_model_id, task2_model_id)
    render_metrics(result)

    tab_analysis, tab_compare, tab_api, tab_notes = st.tabs(
        ["Analisis integrado", "Comparacion de modelos", "Contrato backend", "Notas de despliegue"]
    )

    with tab_analysis:
        render_analysis_tab(result)
    with tab_compare:
        render_comparison_tab(text, task1_model_id)
    with tab_api:
        render_contract_tab(result)
    with tab_notes:
        render_notes_tab()


def render_header() -> None:
    """Render dashboard title and rationale."""

    st.markdown(
        """
        <div class="app-title">Demostrador de analisis de documentos cientificos en espanol</div>
        <div class="app-subtitle">
            Mock de despliegue para explorar segmentacion retorica, deteccion de contribuciones y comparacion de modelos.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, str, str]:
    """Render controls and return selected input/model ids."""

    st.header("Entrada y modelos")
    sample_name = st.selectbox("Texto de prueba", options=list(SAMPLE_TEXTS.keys()), index=0)
    use_sample = st.toggle("Usar texto de prueba", value=True)
    default_text = SAMPLE_TEXTS[sample_name] if use_sample else ""
    text = st.text_area(
        "Texto academico en espanol",
        value=default_text,
        height=320,
        placeholder="Pega aqui un fragmento, resumen o articulo academico...",
    )

    st.divider()
    task1_models = get_model_options("task1")
    task2_models = get_model_options("task2")
    task1_model_name = st.selectbox(
        "Modelo Tarea 1: segmentacion retorica",
        options=[model.name for model in task1_models],
        index=0,
    )
    task2_model_name = st.selectbox(
        "Modelo Tarea 2: contribuciones",
        options=[model.name for model in task2_models],
        index=0,
    )
    task1_model = next(model for model in task1_models if model.name == task1_model_name)
    task2_model = next(model for model in task2_models if model.name == task2_model_name)
    st.caption(f"T1: {task1_model.category} | latencia mock: {task1_model.latency_ms} ms")
    st.caption(f"T2: {task2_model.category} | costo relativo: {task2_model.relative_cost}")
    return text, task1_model.id, task2_model.id


def render_metrics(result: dict) -> None:
    """Render top-level dashboard metrics."""

    summary = result["summary"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fragmentos", summary["segments"])
    col2.metric("Contribuciones", summary["contribution_segments"])
    col3.metric("Confianza T2 prom.", f"{summary['avg_task2_confidence']:.2f}")
    col4.metric("Caracteres", result["input"]["character_count"])


def render_analysis_tab(result: dict) -> None:
    """Render integrated Task 1 and Task 2 predictions."""

    left, right = st.columns([0.68, 0.32], gap="large")
    with left:
        st.subheader("Resultado por fragmento")
        for segment in result["segments"]:
            render_segment_card(segment)
    with right:
        st.subheader("Distribucion retorica")
        distribution = result["summary"]["rhetorical_distribution"]
        chart_data = pd.DataFrame(
            [
                {"Etiqueta": label, "Fragmentos": count, "Nombre": RHETORICAL_LABELS[label]["name"]}
                for label, count in distribution.items()
            ]
        )
        if not chart_data.empty:
            st.bar_chart(chart_data, x="Etiqueta", y="Fragmentos", use_container_width=True)
        st.subheader("Leyenda")
        render_legend()
        st.subheader("Modelos activos")
        st.json(result["models"], expanded=False)


def render_segment_card(segment: dict) -> None:
    """Render one paragraph result as a compact card."""

    task1 = segment["task1"]
    task2 = segment["task2"]
    color = RHETORICAL_LABELS[task1["label"]]["color"]
    contribution_class = "positive" if task2["is_contribution"] else "negative"
    contribution_text = "Contribucion: Si" if task2["is_contribution"] else "Contribucion: No"
    st.markdown(
        f"""
        <div class="segment-card" style="--label-color: {color};">
            <div class="segment-header">
                <span class="badge">P{segment["position"]}</span>
                <span class="badge label-badge">{html.escape(task1["label"])} - {html.escape(task1["label_name"])}</span>
                <span class="badge">Conf. T1 {task1["confidence"]:.2f}</span>
                <span class="badge {contribution_class}">{contribution_text}</span>
                <span class="badge">Conf. T2 {task2["confidence"]:.2f}</span>
            </div>
            <div class="segment-text">{html.escape(segment["text"])}</div>
            <div class="explain">
                <strong>T1:</strong> {html.escape(task1["explanation"])}<br>
                <strong>T2:</strong> {html.escape(task2["evidence"])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_legend() -> None:
    """Render rhetorical label legend."""

    for label, metadata in RHETORICAL_LABELS.items():
        st.markdown(
            f"""
            <div class="legend-row" style="--label-color: {metadata["color"]};">
                <span class="legend-dot"></span>
                <strong>{html.escape(label)}</strong>
                <span>{html.escape(metadata["name"])}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_comparison_tab(text: str, task1_model_id: str) -> None:
    """Render model comparison table for Task 2."""

    st.subheader("Comparacion de arquitecturas para la Tarea 2")
    rows = compare_task2_models(text, task1_model_id)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("La comparacion conserva la misma segmentacion retorica y cambia solo el modelo binario de contribuciones.")


def render_contract_tab(result: dict) -> None:
    """Render backend contract and downloadable mock response."""

    st.subheader("Respuesta JSON esperada del backend")
    json_payload = json.dumps(result, ensure_ascii=False, indent=2)
    st.code(json_payload, language="json")
    st.download_button(
        "Descargar respuesta mock",
        data=json_payload,
        file_name="mock_backend_response.json",
        mime="application/json",
    )


def render_notes_tab() -> None:
    """Render deployment notes inside the dashboard."""

    st.subheader("Por que dashboard")
    st.write(
        "Los documentos del proyecto piden un demostrador para validar modelos en un entorno realista, "
        "comparar arquitecturas frente a la misma entrada y facilitar analisis cualitativo de decisiones. "
        "Por eso el mock se plantea como dashboard operativo: entrada, modelos, resultados, confianza, "
        "comparacion y contrato de backend en un mismo flujo."
    )
    st.subheader("Camino hacia despliegue real")
    st.markdown(
        """
        1. Reemplazar `demo/mock_backend.py` por adaptadores que carguen modelos desde `models/`.
        2. Mantener el mismo contrato JSON para no reescribir la interfaz.
        3. Separar inferencia en un servicio FastAPI si se requiere escalar o exponer API externa.
        4. Registrar latencia, costo y version del modelo para comparar arquitecturas de forma reproducible.
        """
    )


if __name__ == "__main__":
    main()
