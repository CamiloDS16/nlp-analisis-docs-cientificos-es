# Mock de despliegue Streamlit

Este directorio contiene el demostrador interactivo del proyecto **Analisis Automatico de Documentos Cientificos en Espanol**.

El mock esta disenado como dashboard porque los documentos de definicion piden algo mas cercano a una herramienta de analisis que a una pagina informativa: el usuario debe ingresar un texto academico, seleccionar modelos, visualizar predicciones, comparar arquitecturas y revisar confianza/errores sobre la misma entrada.

## Que demuestra

- Flujo integrado de Tarea 1 y Tarea 2.
- Segmentacion retorica por fragmentos.
- Deteccion binaria de contribuciones cientificas explicitas.
- Comparacion entre tres familias de modelos:
  - encoders ajustados para espanol academico;
  - modelos open-weight pequenos usados como decoders;
  - modelos comerciales consumidos via API.
- Contrato JSON esperado para conectar luego un backend real.

## Estructura

```text
demo/
├── app.py              # Interfaz Streamlit tipo dashboard
├── mock_backend.py     # Capa de inferencia simulada y contrato de backend
├── sample_inputs.py    # Textos academicos de prueba
└── README.md           # Instrucciones y notas de despliegue
```

## Ejecucion local

Desde la raiz del repositorio:

```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run demo/app.py
```

Si prefieres no activar el entorno:

```powershell
.\venv\Scripts\python.exe -m streamlit run demo/app.py
```

Streamlit abrira la aplicacion en una URL local similar a:

```text
http://localhost:8501
```

## Como reemplazar el mock por inferencia real

La interfaz consume una unica funcion principal:

```python
analyze_document(text, task1_model_id, task2_model_id)
```

La respuesta debe conservar esta forma general:

```json
{
  "input": {
    "paragraph_count": 6,
    "character_count": 1200
  },
  "models": {
    "task1": {},
    "task2": {}
  },
  "segments": [
    {
      "id": "p1",
      "position": 1,
      "text": "...",
      "task1": {
        "label": "INTRO",
        "label_name": "Introduccion",
        "confidence": 0.82,
        "explanation": "..."
      },
      "task2": {
        "is_contribution": false,
        "label": "No contribucion",
        "confidence": 0.74,
        "evidence": "..."
      }
    }
  ],
  "summary": {}
}
```

Cuando los modelos reales esten listos, la ruta recomendada es:

1. Mantener `app.py` sin cambios grandes.
2. Sustituir las heuristicas de `mock_backend.py` por adaptadores reales.
3. Cargar modelos desde `models/` o desde un servicio externo.
4. Conservar las claves del JSON para que el dashboard siga funcionando.
5. Agregar metricas reales de latencia, costo y version del modelo.

## Notas de alcance

Este mock no entrena ni descarga modelos. Su objetivo es validar la experiencia de despliegue y dejar preparado el contrato de integracion para conectar los clasificadores reales de Tarea 1 y Tarea 2.
