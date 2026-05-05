# A6 — Guía completa: arquitectura, código y ejecución

## Qué es A6

A6 es la actividad del proyecto que evalúa **modelos comerciales de lenguaje grande vía API** para la tarea de segmentación retórica de fragmentos de artículos científicos en español.

El objetivo concreto: dado un fragmento de texto de un paper científico, predecir cuál de las 8 etiquetas retóricas le corresponde:

```
INTRO  BACK  METH  RES  DISC  CONTR  LIM  CONC
```

La evaluación se hace exclusivamente sobre el conjunto EVAL validado manualmente por el equipo, nunca sobre TRAIN_TEST.

---

## Modelo usado

**Gemini 2.0 Flash** (`gemini-2.0-flash-exp`) vía Google AI Studio.

- Free tier: 15 RPM, 1,500 requests/día — suficiente para el EVAL set
- Sin costo económico en free tier
- Se reporta costo equivalente en tier de pago para el análisis A8

El notebook evalúa dos estrategias de prompting sobre el mismo modelo:

| Modo | Descripción |
|---|---|
| Zero-shot | El modelo recibe las definiciones de las 8 etiquetas y el fragmento. Sin ejemplos. |
| Few-shot k=1 | Recibe además 1 ejemplo representativo de cada etiqueta antes del fragmento. |

---

## Archivos que produce el notebook

```
models/a6_api/
├── predictions_zeroshot.csv     # predicciones fila por fila, modo zero-shot
├── predictions_fewshot.csv      # predicciones fila por fila, modo few-shot
├── confusion_matrices.png       # heatmaps comparativos de ambos modos
├── latency_distribution.png     # histogramas de latencia por modo
└── summary_a6.csv               # tabla resumen: F1, costo, latencia por modo
```

Cada archivo de predicciones tiene estas columnas:

```
_idx          índice de la fila en df_eval (sirve para resume/checkpoint)
true_label    etiqueta real (ground truth del EVAL manual)
doc_id        identificador del documento
mode          "zeroshot" o "fewshot"
label         etiqueta predicha por el modelo (None si falló)
confidence    confianza del modelo (0.0–1.0, reportada en el JSON)
tokens_in     tokens de entrada consumidos por este request
tokens_out    tokens de salida consumidos por este request
latency_s     tiempo de respuesta de la API en segundos
error         mensaje de error si el request falló (None si OK)
```

---

## Arquitectura del notebook sección por sección

### Sección 1 — Configuración

```python
import google.generativeai as genai

ROOT = Path("..")
EVAL_PATH   = ROOT / "consolidado_v1_fixed.csv"
RESULTS_DIR = ROOT / "models" / "a6_api"

MODEL = "gemini-2.0-flash-exp"

# Costos de referencia en tier de pago (USD/millón de tokens, abril 2025)
# Con free tier el costo real es $0.00
COST_INPUT_PER_M  = 0.10
COST_OUTPUT_PER_M = 0.40
FREE_TIER = True

TEMPERATURE    = 0
MAX_TOKENS     = 60
MAX_TEXT_WORDS = 700

# Free tier: 15 RPM → mínimo 4 s entre requests
REQUEST_DELAY = 4.1

LABELS = ["INTRO", "BACK", "METH", "RES", "DISC", "CONTR", "LIM", "CONC"]

genai.configure(api_key=api_key)
```

**Por qué `TEMPERATURE=0`**: maximiza el determinismo del modelo. Gemini no tiene parámetro `seed` (a diferencia de GPT-4o), así que `temperature=0` es el único mecanismo de reproducibilidad disponible. Dos ejecuciones sobre el mismo fragmento producen la misma respuesta.

**Por qué `MAX_TEXT_WORDS=700`**: los fragmentos del dataset tienen entre 250 y 1,000 palabras. Truncar a 700 controla el consumo de tokens sin perder información retórica clave, que suele estar concentrada al inicio del fragmento.

**Por qué `REQUEST_DELAY=4.1`**: free tier de Google AI Studio tiene límite de 15 RPM. A 4.1 s entre requests el throughput efectivo es ~14.6 RPM, dejando margen sin caer en rate limit errors (HTTP 429).

**`FREE_TIER=True`**: flag para que el análisis de costo reporte $0 como costo real y el equivalente en tier de pago como referencia comparativa para el informe A8.

---

### Sección 2 — Carga del conjunto de evaluación

```python
df_all   = pd.read_csv(EVAL_PATH)
df_eval  = df_all[df_all["dataset_type"] == "EVAL"]
df_train = df_all[df_all["dataset_type"] == "TRAIN_TEST"]
```

**Archivo fuente**: `consolidado_v1_fixed.csv`. Consolidado del equipo con etiquetas estandarizadas (`metodologia → METH`, `resultados → RES`). Contiene fragmentos de todos los anotadores.

**Por qué separar EVAL de TRAIN_TEST**:
- `df_eval`: conjunto de medición. Solo fragmentos validados manualmente (≥10% del total). Sobre este se calculan Macro F1, Micro F1 y el reporte por etiqueta.
- `df_train`: banco de ejemplos para few-shot. El modelo nunca "aprende" de él — solo se le muestran fragmentos en el contexto del prompt. No se incluye en ninguna métrica.

---

### Sección 3 — Diseño de prompts

#### System prompt

El system prompt tiene tres partes:

1. **Definición del rol**: "Eres un experto en análisis del discurso científico en español con amplio conocimiento en retórica académica."
2. **Definiciones de las 8 etiquetas**: copiadas del enunciado del proyecto, idénticas a los criterios de anotación manual.
3. **Instrucción de output**: `{"label": "<ETIQUETA>", "confidence": <0.0-1.0>}`. Solo JSON.

En Gemini el `system_prompt` se pasa como `system_instruction` en la inicialización del modelo, **no** dentro de los contents de cada request:

```python
gemini_model = genai.GenerativeModel(
    model_name=MODEL,
    generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=60),
    system_instruction=SYSTEM_PROMPT,
)
```

Esto significa que el system prompt se aplica a todos los requests del modelo sin repetirlo en cada llamada.

#### Por qué pedir JSON y no texto libre

- Parseo determinista con `json.loads()`.
- `confidence` captura la certeza implícita del modelo sin tokens adicionales.
- Si el modelo alucina una etiqueta no válida, el parser lo detecta y lo cuenta como fallo.
- `max_output_tokens=60` alcanza para el JSON (`{"label": "INTRO", "confidence": 0.95}` tiene ~15 tokens). Sin esta restricción el modelo puede generar párrafos de explicación.

#### Selección de ejemplos few-shot

```python
def get_fewshot_example(df, label):
    subset = df[df["label"] == label].copy()
    subset["wc"] = subset["text"].str.split().str.len()
    median_wc = subset["wc"].median()
    idx = (subset["wc"] - median_wc).abs().idxmin()
    return " ".join(subset.loc[idx, "text"].split()[:150])
```

Se toma el fragmento de TRAIN_TEST cuyo word count está más cerca de la mediana de su etiqueta. Esto garantiza un ejemplo representativo en longitud, no un outlier. Se trunca a 150 palabras para que los 8 ejemplos no inflen el prompt innecesariamente.

#### Construcción del prompt few-shot para Gemini

Gemini usa un formato de contenidos distinto al de OpenAI:

```python
contents = [
    {"role": "user",  "parts": ["Clasifica: <ejemplo INTRO truncado>"]},
    {"role": "model", "parts": ['{"label": "INTRO", "confidence": 1.0}']},
    {"role": "user",  "parts": ["Clasifica: <ejemplo BACK truncado>"]},
    {"role": "model", "parts": ['{"label": "BACK", "confidence": 1.0}']},
    # ... 6 pares más ...
    {"role": "user",  "parts": ["Clasifica: <fragmento real a predecir>"]},
]
```

Diferencias clave vs OpenAI:
- `"role": "model"` en lugar de `"assistant"`
- `"parts"` (lista) en lugar de `"content"` (string)
- El system_instruction **no** va en contents — ya está en el modelo inicializado

Total por request: 8 pares de ejemplo (16 turns) + 1 query = 17 turns.

---

### Sección 4 — Pipeline de inferencia

#### `parse_label_response`

```python
def parse_label_response(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?\n?", "", raw).replace("```", "").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    parsed = json.loads(match.group())
    label = parsed.get("label", "").strip().upper()
    if label not in LABELS:
        return None
    return {"label": label, "confidence": float(parsed.get("confidence", 0.0))}
```

Maneja tres casos problemáticos:
1. Modelo envuelve JSON en markdown triple-backtick → dos pasos de limpieza (regex + replace).
2. Modelo devuelve texto sin JSON → `re.search` no encuentra `{...}` → devuelve `None`.
3. Modelo inventa una etiqueta → validación `label not in LABELS` → devuelve `None`.

#### `classify_fragment`

```python
def classify_fragment(content, retries=3):
    # content: str (zero-shot) o list (few-shot)
    for attempt in range(retries):
        try:
            t0 = time.time()
            response = gemini_model.generate_content(content)
            result["latency_s"] = time.time() - t0
            result["tokens_in"]  = response.usage_metadata.prompt_token_count
            result["tokens_out"] = response.usage_metadata.candidates_token_count
            raw = response.text.strip()
            parsed = parse_label_response(raw)
            if parsed:
                return {**result, "label": parsed["label"], "confidence": parsed["confidence"]}
        except Exception as e:
            time.sleep(2 ** attempt)   # backoff: 1s, 2s, 4s
    return result  # label=None si falló
```

La función acepta `content` como string (zero-shot) o lista de turns (few-shot). Gemini maneja ambos formatos nativamente con `generate_content()`.

Los tokens se leen de `response.usage_metadata.prompt_token_count` y `candidates_token_count` — disponibles en la respuesta de la API.

#### `run_batch` — checkpoint y resume

```python
if resume and output_path.exists():
    done = pd.read_csv(output_path)
    done_ids = set(done["_idx"].tolist())

for idx, row in tqdm(df.iterrows(), total=len(df)):
    if idx in done_ids:
        continue
    content = build_user_message_zeroshot(row["text"])     # zero-shot
    # o: content = build_fewshot_contents(row["text"], ...)  # few-shot
    result = classify_fragment(content)
    rows.append(result)
    if len(rows) % 50 == 0:
        combined = pd.concat([done, pd.DataFrame(rows)])
        combined.to_csv(output_path, index=False)
        done = combined
        rows = []
```

El checkpoint evita perder trabajo si Colab desconecta la sesión. Al volver a correr, lee el CSV existente y salta los índices ya procesados. Con free tier esto es especialmente importante porque cada sesión de Colab puede desconectarse antes de terminar las ~1,742 predicciones.

---

### Secciones 5 y 6 — Ejecución zero-shot y few-shot

Llaman a `run_batch` con `mode="zeroshot"` y `mode="fewshot"` respectivamente.

**Tiempo estimado con free tier (15 RPM)**:
- ~1,742 requests × 4.1 s = ~2 horas por modo
- Total: ~4 horas para ambos modos
- Dejar Colab corriendo en segundo plano

---

### Sección 7 — Métricas

```python
macro_f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
micro_f1 = f1_score(y_true, y_pred, average="micro", labels=LABELS, zero_division=0)
print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
```

**Macro F1**: promedia el F1 de cada etiqueta sin ponderar por frecuencia. Es la métrica principal del enunciado — el modelo tiene que funcionar bien en etiquetas minoritarias (LIM, CONTR) para tener buen Macro F1.

**Micro F1**: pondera por frecuencia de cada etiqueta. Si Macro < Micro, el modelo falla desproporcionalmente en las etiquetas pequeñas.

**`classification_report`**: tabla con precision, recall, F1 y support por etiqueta. Va directo al informe A7 comparativo.

**Matrices de confusión**: normalizadas por fila (colores = proporción) con valores absolutos en las celdas. Se grafican ambos modos lado a lado para ver si few-shot reduce confusiones específicas (DISC↔CONC, INTRO↔BACK).

---

### Sección 8 — Análisis de costo y latencia (A8)

```python
equiv_cost = (total_in / 1e6) * COST_INPUT_PER_M + (total_out / 1e6) * COST_OUTPUT_PER_M
real_cost  = 0.0 if FREE_TIER else equiv_cost
```

Se reporta:
- **Costo real**: $0.00 con free tier
- **Costo equivalente en tier de pago**: para comparar con GPT-4o-mini, Claude, etc. en el informe A8
- **Latencia media y p95**: distribución de tiempos de respuesta por modo

Los tokens reportados son los reales de la API (`usage_metadata`), no estimados.

---

### Sección 9 — Análisis de errores

`error_analysis` reporta pares de confusión más frecuentes y tasa de error por etiqueta.

`show_error_examples` muestra los textos reales de fragmentos mal clasificados para diagnosticar si el error es del modelo o si el fragmento era genuinamente ambiguo.

---

### Sección 10 — Tabla resumen

`summary_a6.csv` tiene una fila por estrategia con Macro F1, Micro F1, costo real, costo equivalente, latencia media, requests válidos/fallidos. Esta tabla va al informe comparativo A7/A8 junto con los números de SciBERT (Jesús/Mateo) y TF-IDF (Sergio).

---

## Cómo ejecutar en Google Colab paso a paso

### 1. Obtener la API key de Gemini (gratis)

- Ve a [aistudio.google.com](https://aistudio.google.com)
- Clic en "Get API Key" → "Create API key in new project"
- Copia la key (empieza con `AIza...`)

### 2. Subir archivos a Google Drive

Sube al menos `consolidado_v1_fixed.csv` a una carpeta en Drive, por ejemplo: `Mi unidad/nlp-proyecto/`

### 3. Abrir el notebook en Colab

Sube `notebooks/04_a6_api_classification.ipynb` a Colab o ábrelo desde Drive.

### 4. Instalar dependencias

Descomenta y corre la primera celda:

```python
!pip install google-generativeai pandas scikit-learn matplotlib seaborn tqdm
```

### 5. Agregar la API key en Colab Secrets

- Ícono de llave en el panel izquierdo → "Add new secret"
- Nombre: `GOOGLE_API_KEY`, Valor: tu key de Gemini
- El notebook la lee automáticamente — no necesitas cambiar ninguna línea de código

### 6. Montar Drive y ajustar rutas

Agrega una celda al inicio del notebook (antes de la celda de config):

```python
from google.colab import drive
drive.mount('/content/drive')
```

Luego en la sección de config cambia `ROOT`:

```python
ROOT = Path("/content/drive/MyDrive/nlp-proyecto")
```

Las demás rutas (`EVAL_PATH`, `RESULTS_DIR`) se construyen automáticamente sobre `ROOT`.

### 7. Correr todo en orden

Las secciones 5 y 6 son las que llaman a la API. Con free tier (15 RPM) cada una toma ~2 horas. Dejar Colab abierto o usar la opción de sesión larga.

**Si Colab desconecta la sesión**:
1. Vuelve a montar Drive
2. Corre de nuevo las celdas de config, imports, data loading, y prompts
3. Corre de nuevo la celda de `run_batch` correspondiente — lee el checkpoint y retoma desde donde quedó sin reprocesar nada

### 8. Descargar resultados

Los archivos ya están en Drive en `nlp-proyecto/models/a6_api/`. Descarga `summary_a6.csv`, `confusion_matrices.png` y `latency_distribution.png` para el informe.

---

## Estimación de tiempo y costo

| Modelo | Costo real | Costo equiv. tier pago | Tiempo (~1.7K requests × 2 modos) |
|---|---|---|---|
| `gemini-2.0-flash-exp` (free tier) | **$0** | ~$0.40 | ~4 horas |
| `gemini-2.0-flash` (paid) | ~$0.40 | ~$0.40 | ~20 min (1000 RPM) |
| `gpt-4o-mini` | ~$0.50 | ~$0.50 | ~10 min (500 RPM) |
| `gpt-4o` | ~$5.00 | ~$5.00 | ~10 min |

---

## Métricas que produce para el informe

Para **A7** (comparativa de modelos):
- Macro F1 por modo (zero-shot, few-shot)
- Micro F1 por modo
- Precision / Recall / F1 por etiqueta (del `classification_report`)
- Matrices de confusión comparativas

Para **A8** (análisis de costo y eficiencia):
- Costo real USD (free tier = $0)
- Costo equivalente en tier de pago para comparación justa
- Costo por 1,000 documentos
- Latencia media y p95 por modo
- Número de requests fallidos

---

## Lineamientos del enunciado que este notebook cumple

| Lineamiento | Cómo se cumple |
|---|---|
| Evaluación solo sobre EVAL | `df_eval = df_all[df_all["dataset_type"] == "EVAL"]` |
| No solapamiento EVAL/TRAIN_TEST por documento | Garantizado por `consolidado_v1_fixed.csv` |
| Dos estrategias de prompting | Secciones 5 (zero-shot) y 6 (few-shot k=1) |
| Macro F1 como métrica principal | `f1_score(..., average="macro")` |
| Análisis de costo (A8) | Sección 8: tokens reales × precio por millón |
| Reproducibilidad | `temperature=0` (Gemini no tiene parámetro seed) |
| Modelo comercial vía API | Gemini 2.0 Flash — Google AI Studio |
