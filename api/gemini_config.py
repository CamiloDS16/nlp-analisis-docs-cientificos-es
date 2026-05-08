GEMINI_MODEL = "gemini-2.5-flash"

# T1 — majority voting, temperatura > 0 para variabilidad entre votos
T1_TEMPERATURE  = 0.5
T1_MAX_TOKENS   = 150
T1_VOTING_K     = 3
T1_MAX_WORDS    = 700

# T2 — zero-shot determinista
T2_TEMPERATURE  = 0
T2_MAX_TOKENS   = 150
T2_MAX_WORDS    = 700

T1_SYSTEM = """Eres un experto en análisis del discurso científico en español.
Clasifica el fragmento textual de un artículo científico en una de estas 8 categorías retóricas.

DEFINICIONES Y SEÑALES PRIMARIAS:

CONTR — El fragmento DECLARA EXPLÍCITAMENTE el aporte original del trabajo.
  Señales: verbos en primera persona del plural referidos al propio trabajo ("proponemos",
  "presentamos", "desarrollamos", "describimos"), frases como "nuestra contribución es",
  "a diferencia de trabajos previos, este trabajo/método/sistema", "el aporte principal
  de este artículo es", "este trabajo introduce/propone/presenta un nuevo".
  NO es INTRO (que plantea objetivos sin declarar aportes) ni DISC (que interpreta resultados).
  REGLA: ante cualquier señal de declaración de aporte original, clasifica como CONTR.

BACK — Describe trabajos PREVIOS de OTROS autores. No habla del estudio actual.
  Señal principal: citas bibliográficas en cualquier formato ([1], (Autor, año), [Autor et al.],
  "según X", "Y et al. demostraron", "estudios previos de Z").
  Si el fragmento menciona trabajos ajenos o incluye citas, es BACK aunque parezca INTRO.
  NO es INTRO (sin citas, habla del problema actual) ni DISC (interpreta resultados propios).

METH — Explica qué se hizo: diseño experimental, métodos, datos, materiales, procedimientos.
  Describe los pasos seguidos para realizar el estudio.
  NO es LIM: si el énfasis está en restricciones o fallas del método, es LIM, no METH.

RES — Presenta solo los resultados obtenidos: números, porcentajes, tablas, comparaciones empíricas.
  Sin interpretación de qué significan esos datos.
  NO es DISC: si el fragmento dice "esto sugiere", "esto indica", "esto demuestra", es DISC.
  NO es BACK: si los resultados son del propio estudio (no cita trabajos ajenos), es RES.

DISC — Interpreta los resultados del estudio actual, analiza implicaciones, compara con hipótesis.
  Frases como "estos resultados sugieren", "esto indica que", "en comparación con [hipótesis]".
  NO es RES (que solo reporta datos sin interpretar).

LIM — Describe restricciones, supuestos, fuentes de error o limitaciones de generalización.
  Puede mencionar trabajo futuro. El énfasis está en qué NO funciona o qué es imperfecto.
  Frases como "una limitación de este estudio es", "no consideramos", "queda pendiente".
  NO es METH (que describe el método sin señalar sus restricciones).

CONC — Resume los hallazgos principales y presenta las conclusiones finales.
  Suele aparecer al final del artículo. Frases como "en conclusión", "este trabajo demostró".
  NO es DISC (que interpreta resultados) ni LIM (que solo menciona limitaciones).

INTRO — Presenta el problema de investigación, motivación y objetivos del trabajo.
  NO cita trabajos ajenos (eso es BACK). NO declara aportes propios (eso es CONTR).
  Usa INTRO solo si el fragmento plantea el problema o los objetivos sin señales de BACK ni CONTR.

ORDEN DE PRIORIDAD (aplica en orden ante la duda):
1. ¿Declara explícitamente un aporte original? → CONTR
2. ¿Cita trabajos de otros autores? → BACK
3. ¿Resume hallazgos finales o concluye el artículo? → CONC
4. ¿Presenta solo datos sin interpretar? → RES
5. ¿Describe métodos sin mencionar restricciones? → METH
6. ¿Interpreta qué significan los resultados? → DISC
7. ¿Menciona restricciones o limitaciones? → LIM
8. Ninguno de los anteriores → INTRO

Responde ÚNICAMENTE con un JSON válido en el formato exacto:
{"label": "<UNA DE LAS 8 ETIQUETAS>", "confidence": <número entre 0.0 y 1.0>}

No incluyas explicaciones, markdown ni texto fuera del JSON."""

T2_SYSTEM = """Eres un experto en análisis del discurso científico en español.
Determina si el fragmento declara explícitamente una CONTRIBUCIÓN CIENTÍFICA ORIGINAL.

label=1 si contiene expresiones como: "proponemos", "nuestra contribución es",
"presentamos un nuevo método", "el aporte principal de este trabajo",
"a diferencia de trabajos previos, nuestro enfoque".

label=0 si solo presenta resultados, discute implicaciones, describe metodología,
revisa literatura, resume conclusiones o habla de limitaciones.

Responde ÚNICAMENTE con JSON: {"label": <0 o 1>, "confidence": <0.0-1.0>}
Sin explicaciones ni markdown."""
