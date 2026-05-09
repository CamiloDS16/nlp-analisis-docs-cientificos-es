# Análisis automático de documentos científicos en español

Proyecto de grado — Maestría en Inteligencia Artificial, Universidad de los Andes.

El español académico latinoamericano carece de recursos NLP especializados para análisis del discurso científico. Este proyecto construye un corpus anotado y evalúa cuatro familias de modelos sobre dos tareas de clasificación de fragmentos textuales, con un sistema desplegado que permite comparar arquitecturas sobre texto de entrada libre.

---

## Tareas

**T1 — Clasificación retórica:** dado un fragmento, asignarle una de ocho funciones discursivas.

| Etiqueta | Función |
|----------|---------|
| `INTRO`  | Introducción |
| `BACK`   | Antecedentes |
| `METH`   | Metodología |
| `RES`    | Resultados |
| `DISC`   | Discusión |
| `CONTR`  | Contribución científica |
| `LIM`    | Limitaciones / trabajo futuro |
| `CONC`   | Conclusiones |

**T2 — Detección de contribución:** clasificación binaria — el fragmento declara explícitamente una contribución científica original (1) o no (0).

---

## Corpus

Construido a partir del repositorio CORE (~1.8M documentos en español). Los fragmentos se definen a nivel de párrafo. El etiquetado es híbrido: ~10% anotación manual por cuatro anotadores con guía de anotación, ~90% etiquetado automático por heurísticas. Acuerdo inter-anotador: κ = 0.76 (Cohen's Kappa). Dataset final: >16.000 fragmentos etiquetados, particionado en TRAIN / TEST / EVAL.

---

## Modelos evaluados

Se evalúan cuatro familias bajo la misma partición EVAL:

- **Baseline:** modelos tradicionales (TF-IDF)
- **Encoder fine-tuned:** modelos transformer ajustados al dominio
- **Open-weight:** modelos decoder de mediana escala
- **LLM comercial:** modelos de gran escala vía API con prompt engineering

Resultados en `notebooks/` — un notebook por familia de modelos.

---

## Sistema desplegado

La API (FastAPI) expone tres slots de modelo intercambiables — LLM comercial, encoder fine-tuned y open-weight — sobre los mismos endpoints. La interfaz de demostración (Streamlit) permite clasificar texto libre y comparar las tres arquitecturas en paralelo sobre el mismo input.

```
GET  /health
GET  /models/{task}
POST /analyze     — segmenta el texto y corre T1 + T2
POST /compare     — fija T1, compara los tres modelos de T2
```

Para instrucciones de despliegue en EC2 ver `docker/README.md`.  
Para la interfaz de demostración ver `demo/README.md`.

---

## Estructura del repositorio

```
api/              backend FastAPI — inferencia y prompts centralizados
demo/             interfaz Streamlit
docker/           configuración de despliegue
notebooks/        evaluación experimental T1 y T2
docs/             informe del proyecto
data/             datasets (excluidos de git)
models/           pesos fine-tuned (excluidos de git)
label_studio/     tareas de anotación Cohen
```
