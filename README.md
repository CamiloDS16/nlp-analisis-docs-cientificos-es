# Análisis Automático de Documentos Científicos en Español

Proyecto de tesis de maestría — Universidad de los Andes  
Construcción de un corpus anotado y modelos de clasificación de secciones
en artículos científicos escritos en español.

---

## Descripción

Este proyecto desarrolla la infraestructura de datos y los modelos necesarios
para identificar y clasificar automáticamente las secciones retóricas de
documentos científicos en español (introducción, metodología, resultados, etc.).

El corpus proviene de ~1.342.100 archivos `.txt` almacenados en Google Drive
compartido del equipo. El flujo completo incluye:

1. Descarga y partición del corpus por miembro del equipo
2. Fragmentación y etiquetado automático por heurísticas
3. Validación humana de una muestra (10% por etiqueta)
4. Entrenamiento y evaluación de clasificadores

---

## Asignación de etiquetas por miembro

| Miembro | Etiquetas asignadas          |
|---------|------------------------------|
| Jesus   | `INTRO`, `BACK`              |
| Camilo  | `LIM`, `CONC`                |
| Mateo   | `METH`, `RES`                |
| Sergio  | `DISC`, `CONTR`              |

**Definición de etiquetas:**

| Etiqueta | Significado                         |
|----------|-------------------------------------|
| `INTRO`  | Introducción                        |
| `BACK`   | Antecedentes / Marco teórico        |
| `METH`   | Metodología                         |
| `RES`    | Resultados                          |
| `DISC`   | Discusión                           |
| `CONTR`  | Contribuciones                      |
| `LIM`    | Limitaciones / Trabajo futuro       |
| `CONC`   | Conclusiones                        |

---

## Estructura del repositorio

```
nlp-analisis-docs-cientificos-es/
├── data/
│   ├── particiones.csv          # Asignación de documentos por miembro
│   ├── raw/                     # Punteros .dvc (los .parquet viven en Drive)
│   └── processed/
│       ├── task1/               # Clasificación de secciones
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── task2/               # Tarea secundaria
│           ├── train/
│           └── test/
├── notebooks/
│   ├── 01_carga_corpus.ipynb    # Descarga y filtra corpus desde Drive
│   └── 02_etiquetado_task1.ipynb # Etiquetado automático + muestra humana
├── src/
│   ├── __init__.py
│   ├── partition_dataset.py     # Lógica de partición train/val/test
│   ├── preprocessing.py         # Limpieza y normalización de texto
│   ├── etiquetado_heuristico.py # Patrones regex por etiqueta
│   ├── desacoplamiento.py       # Separación de capas de datos
│   └── setup_dvc.sh             # Script de configuración de DVC
├── models/                      # Modelos entrenados (ignorados por git)
├── demo/                        # Aplicación de demostración
├── .gitignore
├── .dvcignore
├── dvc.yaml                     # Pipeline de DVC
└── requirements.txt
```

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
source venv/bin/activate          # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar DVC

```bash
bash src/setup_dvc.sh
```

DVC está configurado con Google Drive como remote. La primera vez que hagas
`dvc pull` o `dvc push`, te pedirá autenticación con tu cuenta de Google.

### 4. Descargar datos (si ya existen punteros .dvc)

```bash
dvc pull
```

---

## Flujo de trabajo para cada miembro

1. Abrir `notebooks/01_carga_corpus.ipynb` en Google Colab
2. Cambiar `NOMBRE = "TuNombre"` en la celda de configuración
3. Ejecutar todas las celdas → genera `{nombre}_corpus.parquet` en Drive
4. Abrir `notebooks/02_etiquetado_task1.ipynb` en Colab
5. Cambiar `NOMBRE` y `ETIQUETAS_ASIGNADAS` según la tabla de arriba
6. Ejecutar todas las celdas → genera dataset etiquetado y muestra de validación

---

## Versiones principales

- Python 3.10+
- Transformers (Hugging Face)
- DVC con remote en Google Drive
