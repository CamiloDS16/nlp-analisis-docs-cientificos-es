# Análisis Automático de Documentos Científicos en Español

Proyecto de tesis de maestría — Universidad de los Andes, Colombia.

La mayoría de herramientas de PLN para análisis de literatura científica están diseñadas para el inglés. El español carece de recursos especializados para tareas como la segmentación retórica de artículos o la identificación automática de contribuciones, lo que limita la capacidad de los investigadores hispanohablantes para procesar su propia producción científica a escala.

Este proyecto construye un pipeline completo de PLN sobre un corpus de ~1.342.100 documentos científicos en español extraídos de la API de CORE. El sistema aborda dos tareas: clasificar la función retórica de fragmentos de texto (Tarea 1) y detectar automáticamente si un fragmento contiene una contribución científica explícita (Tarea 2). Los modelos se comparan en tres enfoques: encoders fine-tuneados (SciBETO-large), LLMs open-weight de 1–8B en modo inferencia, y modelos comerciales vía API.

---

## Tareas del proyecto

### Tarea 1 — Segmentación retórica

Dado un fragmento de entre 250 y 1.000 palabras extraído de un artículo científico en español, el modelo debe clasificar su función retórica en una de ocho categorías. Estas categorías representan las secciones convencionales del discurso científico académico. El dataset de entrenamiento requiere mínimo 2.000 fragmentos por etiqueta.

| Etiqueta | Descripción                                          |
|----------|------------------------------------------------------|
| INTRO    | Introducción, motivación y objetivos del trabajo     |
| BACK     | Antecedentes, marco teórico y estado del arte        |
| METH     | Metodología, diseño experimental y procedimientos    |
| RES      | Resultados, métricas y evaluación empírica           |
| DISC     | Discusión, interpretación y comparación con trabajos previos |
| CONTR    | Contribuciones originales del trabajo                |
| LIM      | Limitaciones del estudio y líneas de trabajo futuro  |
| CONC     | Conclusiones y síntesis final                        |

### Tarea 2 — Extracción de contribuciones científicas

Clasificación binaria: dado un fragmento, determinar si contiene o no una contribución científica explícita. El dataset requiere mínimo 1.000 ejemplos positivos y 1.000 negativos. Esta tarea utiliza como entrada la segmentación producida por la Tarea 1, particularmente los fragmentos clasificados como CONTR.

---

## Asignación de etiquetas por miembro

| Miembro | Etiquetas   |
|---------|-------------|
| Jesus   | INTRO, BACK |
| Camilo  | LIM, CONC   |
| Mateo   | METH, RES   |
| Sergio  | DISC, CONTR |

---

## Estructura del repositorio

```
nlp-analisis-docs-cientificos-es/
├── data/
│   ├── particiones.csv              # índices de documentos asignados por miembro (seed=42)
│   ├── raw/                         # punteros .dvc — los parquets reales viven en Drive
│   └── processed/
│       ├── task1/                   # datasets de segmentación retórica
│       │   ├── train/               # fragmentos etiquetados para entrenamiento
│       │   ├── val/                 # validación durante entrenamiento
│       │   └── test/                # evaluación final — solo muestras con anotación humana
│       └── task2/                   # datasets de extracción de contribuciones
│           ├── train/
│           └── test/
├── notebooks/
│   ├── CARGA_TXT_MAIA_PROJECT.ipynb                        # Fase 2: descarga y partición del corpus por miembro
│   ├── 02_etiquetado_task1.ipynb                           # Fase 3–5: fragmentación, etiquetado (LIM/CONC), desacople y validación
│   └── task1_disc_contr_automatic_annotation_notebook_v2.ipynb  # Fase 3–5: etiquetado (DISC/CONTR), Sergio
├── src/
│   └── seleccion-aleatoria.py       # partición del corpus por miembro con semilla fija
├── models/                          # modelos entrenados (ignorados por git)
├── demo/                            # demostrador interactivo (pendiente)
├── dvc.yaml                         # pipeline DVC
├── .dvcignore
├── .gitignore
└── requirements.txt
```

---

## Flujo de trabajo completo

Los datos no están en el repositorio — viven en Google Drive. DVC gestiona los punteros `.dvc` que permiten reproducir cualquier versión del dataset sin almacenar binarios en git.

| Fase | Descripción | Notebook o script |
|------|-------------|-------------------|
| 1. Partición del corpus | Asignar documentos a cada miembro con semilla fija | `src/seleccion-aleatoria.py` |
| 2. Carga del corpus | Descargar los .txt asignados y generar parquet por miembro | `CARGA_TXT_MAIA_PROJECT.ipynb` |
| 3. Fragmentación y etiquetado | Dividir artículos en fragmentos de 250–1.000 palabras y asignar etiqueta por patrones regex sobre el encabezado (Dataset A) | `02_etiquetado_task1.ipynb` / `task1_disc_contr...ipynb` |
| 4. Desacople de patrones | Neutralizar los patrones léxicos usados para etiquetar para evitar data leakage (Dataset B) | mismo notebook, celda de desacople |
| 5. Validación humana | Revisar manualmente 200 fragmentos por etiqueta usando la planilla CSV generada | planilla exportada por el notebook |
| 6. Acuerdo interanotador | Calcular Cohen's Kappa / Fleiss' Kappa sobre la muestra revisada | pendiente |
| 7. Entrenamiento | Fine-tuning de SciBETO-large sobre Dataset A y Dataset B | pendiente |
| 8. Evaluación comparativa | Comparar encoders fine-tuneados, LLMs open-weight (1–8B) y modelos comerciales vía API | pendiente |
| 9. Demostrador interactivo | Aplicación que integra ambas tareas con selección de modelo y visualización por colores | `demo/` — pendiente |

---

## Setup

### 1. Clonar el repositorio

```bash
git clone https://github.com/CamiloDS16/nlp-analisis-docs-cientificos-es.git
cd nlp-analisis-docs-cientificos-es
```

### 2. Instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar DVC

Los datos están en Google Drive compartido del equipo. DVC conecta el repositorio con ese remote.

```bash
bash src/setup_dvc.sh
```

La primera vez que corras `dvc pull` pedirá autenticación con tu cuenta de Google.

### 4. Descargar datos

```bash
dvc pull
```

---

## Estado del proyecto

| Fase                          | Estado      |
|-------------------------------|-------------|
| Partición del corpus          | Completado  |
| Carga del corpus              | En progreso |
| Etiquetado Tarea 1            | En progreso |
| Validación humana             | Pendiente   |
| Entrenamiento de modelos      | Pendiente   |
| Evaluación comparativa        | Pendiente   |
| Tarea 2                       | Pendiente   |
| Demostrador interactivo       | Pendiente   |

---

## Stack tecnológico

**Modelos:** Hugging Face Transformers, SciBETO-large, Llama 3 (inferencia), GPT y Gemini vía API

**Datos:** DVC, Google Drive, CORE API

**Procesamiento:** Python 3.10+, pandas, PyArrow, tqdm

**Entrenamiento:** PyTorch, scikit-learn

**Despliegue:** Docker, Amazon ECS

---

## Equipo

- Jesús Vilardi 
- Camilo Durango
- Mateo Gúzman
- Sergio Angarita
