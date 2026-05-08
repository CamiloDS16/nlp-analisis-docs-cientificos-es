"""Sample academic texts for the Streamlit deployment mock."""

SAMPLE_TEXTS = {
    "Articulo PLN cientifico": """El crecimiento de la literatura academica en espanol ha incrementado la necesidad de herramientas automaticas para organizar y analizar documentos cientificos. Este trabajo aborda la segmentacion retorica y la identificacion de contribuciones explicitas en articulos academicos escritos en espanol.

Los trabajos previos se han concentrado principalmente en recursos para ingles, mientras que los corpus anotados para espanol cientifico siguen siendo limitados. Investigaciones recientes en modelos de lenguaje muestran avances relevantes, pero aun existe una brecha en tareas especializadas de analisis documental.

La metodologia propuesta construye un corpus balanceado a partir de documentos extraidos de CORE. Cada texto se normaliza, se divide en fragmentos coherentes y se etiqueta mediante heuristicas iniciales que luego son revisadas por anotadores humanos.

Presentamos como contribucion principal un marco integrado que combina segmentacion retorica y deteccion de aportes cientificos. El sistema permite comparar clasificadores basados en encoders, modelos open-weight y modelos comerciales via API sobre una misma entrada textual.

Los resultados preliminares muestran que el uso de informacion retorica mejora la deteccion de contribuciones frente a una clasificacion aislada por parrafo. En particular, los fragmentos etiquetados como metodologia, resultados y discusion concentran la mayor proporcion de aportes explicitos.

Una limitacion del estudio es la dependencia de anotaciones manuales para validar una parte del corpus. Como trabajo futuro, se ampliara la evaluacion a dominios cientificos adicionales y se estudiara la estabilidad de los modelos ante textos mas extensos.""",
    "Resumen de tesis": """Esta investigacion estudia el analisis automatico de documentos cientificos en espanol mediante tecnicas de procesamiento de lenguaje natural. El objetivo es apoyar la organizacion de literatura academica y facilitar la identificacion de aportes relevantes.

Se propone un pipeline compuesto por dos tareas. La primera clasifica fragmentos textuales en categorias retoricas como introduccion, antecedentes, metodologia, resultados, discusion, contribuciones, limitaciones y conclusiones. La segunda determina si cada parrafo contiene una contribucion cientifica explicita.

El enfoque combina modelos transformer ajustados, modelos pequenos de pesos abiertos y modelos comerciales consultados via API. Esta comparacion permite analizar desempeno, costo computacional, estabilidad y utilidad practica para escenarios reales de investigacion.

En conclusion, el proyecto busca entregar datos anotados, modelos evaluados y un demostrador interactivo que permita explorar predicciones, revisar errores y comparar arquitecturas de PLN para el dominio cientifico en espanol.""",
    "Fragmento metodologico": """Para construir el conjunto de datos se seleccionaron documentos academicos con estructura reconocible. El texto fue limpiado para remover referencias, tablas y elementos no textuales antes de ser segmentado en unidades continuas.

Cada fragmento fue procesado mediante reglas basadas en encabezados, posicion relativa y patrones lexicos. Posteriormente se entrenaron clasificadores supervisados con representaciones transformer y se evaluaron con precision, recall y F1 macro.

El analisis de errores permitio identificar falsos positivos en parrafos que describen resultados sin novedad explicita. Estos casos fueron revisados para ajustar la guia de anotacion y mejorar la consistencia entre anotadores.""",
}
