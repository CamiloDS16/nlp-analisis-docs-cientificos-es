# Clasificación retórica de documentos científicos en español

Proyecto de grado — Maestría en Inteligencia Artificial, Universidad de los Andes.

Corpus anotado y modelos de clasificación para identificar la función retórica de fragmentos en artículos científicos en español latinoamericano.

---

## Tareas

**Tarea 1 — Clasificación retórica (8 clases)**

Dado un fragmento de texto, clasificarlo en una de ocho secciones retóricas:

| Etiqueta | Sección |
|----------|---------|
| `INTRO`  | Introducción |
| `BACK`   | Antecedentes / Marco teórico |
| `METH`   | Metodología |
| `RES`    | Resultados |
| `DISC`   | Discusión |
| `CONTR`  | Contribución científica |
| `LIM`    | Limitaciones / Trabajo futuro |
| `CONC`   | Conclusiones |

**Tarea 2 — Detección de contribución (binaria)**

Dado un fragmento, determinar si declara explícitamente una contribución científica original (1) o no (0).

---

## Dataset

~17.000 fragmentos extraídos de artículos y tesis de universidades latinoamericanas. Anotación humana distribuida entre cuatro miembros del equipo con validación por kappa de Cohen.

| Miembro | Etiquetas T1 |
|---------|-------------|
| Jesús   | `INTRO`, `BACK` |
| Camilo  | `LIM`, `CONC` |
| Mateo   | `METH`, `RES` |
| Sergio  | `DISC`, `CONTR` |

Dataset consolidado: `data/Dataset_consolidado_final_v4.csv` — particiones `TRAIN`, `TEST`, `EVAL`.

---

## Modelos evaluados

| Slot | Modelo T1 | Macro F1 T1 | Modelo T2 |
|------|-----------|-------------|-----------|
| LLM comercial | Gemini 2.5 Flash, few-shot k=3 | 0.497 | Gemini 2.5 Flash, zero-shot |
| Encoder fine-tuned | SciBETO (Sergio) | — | Pendiente |
| Open-weight | LLaMA 3 vía Ollama | — | Pendiente |

La estrategia few-shot k=3 (majority voting, temperatura 0.5) se comparó contra zero-shot (temperatura 0) sobre 1699 fragmentos EVAL. Resultados en `notebooks/04_a6_api_classification_v6.ipynb`.

---

## Estructura

```
api/                  FastAPI — inferencia T1 + T2, tres slots de modelo
  gemini_config.py    prompts y parámetros centralizados
  fewshot_examples.json  8 ejemplos few-shot (uno por etiqueta, trazables a TRAIN)
demo/                 Streamlit — interfaz de demostración (Sergio)
docker/               Dockerfile + docker-compose.yml + README de despliegue
notebooks/
  04_a6_api_classification_v6.ipynb   evaluación T1: zero-shot vs few-shot k=3
  06_task2_gemini_classifier.ipynb    evaluación T2: clasificación binaria CONTR
models/               pesos fine-tuned (excluidos de git, transferir por SCP)
label_studio/         tareas de anotación por miembro para kappa Cohen
data/
  Dataset_consolidado_final_v4.csv   dataset T1 consolidado
  DATASET_TAREA_2_CONSOLIDADO_FINAL_ORIGINAL.xlsx  dataset T2
```

---

## Despliegue

EC2 t3.small (AWS Academy), Docker Compose V2. Ver `docker/README.md`.

```bash
cd docker
docker compose up --build -d
curl -s localhost:8000/health
```

Frontend en `http://<IP>:8501`, API en `http://<IP>:8000`.

---

## Notebooks en Colab

Los notebooks de evaluación corren en Google Colab con acceso a Drive.

**T1 (`04_a6_api_classification_v6.ipynb`):**
- Subir `data/Dataset_consolidado_final_v4.csv` y `api/fewshot_examples.json` a la raíz de Drive
- Agregar `GOOGLE_API_KEY` en Colab Secrets con el toggle activo

**T2 (`06_task2_gemini_classifier.ipynb`):**
- Subir `DATASET_TAREA_2_CONSOLIDADO_FINAL_ORIGINAL.xlsx` a la raíz de Drive
- Misma API key
