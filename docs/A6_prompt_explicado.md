# A6 — El prompt: texto completo y justificación de cada decisión

## Contexto: por qué hay un prompt en este experimento

Gemini 2.0 Flash es un modelo de lenguaje general. No fue entrenado específicamente para clasificar retórica académica en español. El prompt es la única forma de orientarlo hacia la tarea sin modificar sus pesos (lo que sería fine-tuning, que es A4).

La calidad del prompt determina directamente la calidad de las predicciones. Un prompt mal escrito puede hacer que un modelo muy capaz clasifique peor que uno mediocre con buen prompt.

---

## El system prompt completo

```
Eres un experto en análisis del discurso científico en español con amplio conocimiento
en retórica académica. Tu tarea es clasificar fragmentos textuales de artículos
científicos según su función retórica.

Las ocho categorías posibles y sus definiciones son:

INTRO: Presenta el problema de investigación, su motivación, los objetivos del trabajo
y, en algunos casos, una descripción general del enfoque propuesto.

BACK: Describe el estado del arte, trabajos previos relevantes y el contexto teórico
en el que se enmarca la investigación. Suele citar autores externos al estudio actual.

METH: Explica el diseño experimental, los métodos, modelos, datos, materiales y
procedimientos utilizados para desarrollar el estudio.

RES: Presenta los resultados obtenidos a partir de los experimentos, análisis empíricos
o evaluaciones realizadas, generalmente sin interpretación extensiva.

DISC: Interpreta los resultados del estudio actual, analiza sus implicaciones, los
compara con trabajos previos y discute su relevancia.

CONTR: Identifica explícitamente los aportes del trabajo: métodos propuestos, hallazgos
principales o avances conceptuales novedosos.

LIM: Describe restricciones del enfoque, supuestos adoptados, posibles fuentes de error
o aspectos que limitan la generalización de los resultados. Puede incluir trabajo futuro.

CONC: Resume los principales hallazgos del trabajo y presenta las conclusiones finales
o líneas de trabajo futuro.

Responde ÚNICAMENTE con un JSON válido en el formato exacto:
{"label": "<UNA DE LAS 8 ETIQUETAS>", "confidence": <número entre 0.0 y 1.0>}

No incluyas explicaciones, markdown ni texto fuera del JSON.
```

---

## Cómo se entrega el system prompt a Gemini

En Gemini el system prompt **no va dentro de los mensajes** — va como `system_instruction` al inicializar el modelo:

```python
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=60),
    system_instruction=SYSTEM_PROMPT,   # ← aquí, una sola vez
)
```

Esto es diferente a OpenAI, donde el system prompt se incluye como el primer mensaje de cada request (`{"role": "system", "content": ...}`). Con Gemini, el `system_instruction` se aplica automáticamente a todos los `generate_content()` llamados desde ese objeto modelo.

---

## Análisis línea por línea

### "Eres un experto en análisis del discurso científico en español con amplio conocimiento en retórica académica."

**Qué hace**: define el rol del modelo antes de darle la tarea.

**Por qué importa**: los LLMs son sensibles al framing del rol. Cuando se establece un rol experto relevante, el modelo activa patrones de lenguaje asociados a ese dominio — terminología académica, razonamiento sobre estructura textual, criterios de análisis de discurso. Sin esto, el modelo responde desde un rol genérico y produce clasificaciones más superficiales.

**Por qué "en español"**: los fragmentos están en español y las etiquetas del equipo fueron definidas para textos en español. Mencionar el idioma ancla al modelo al dominio lingüístico correcto y reduce la probabilidad de que mezcle criterios de convenciones académicas en inglés.

---

### Las 8 definiciones de etiquetas

**Qué hace**: le dice al modelo exactamente qué significa cada categoría para este proyecto.

**Por qué son necesarias**: Gemini conoce los términos académicos generales (introducción, metodología, conclusiones), pero etiquetas como `CONTR` y `LIM` tienen significados específicos en este esquema que no son obvios. Sin definiciones, el modelo usaría su propia interpretación, que puede no coincidir con los criterios de anotación del equipo.

**Por qué se tomaron del enunciado y no se reformularon**: para que sean exactamente los mismos criterios con que los anotadores humanos construyeron el EVAL set. Si las definiciones del prompt difieren de las definiciones con que se anotó, el modelo y los humanos están usando estándares distintos y las métricas no miden lo que parecen medir.

**Detalles que distinguen etiquetas similares**:

- `RES` → "generalmente sin interpretación extensiva" — diferencia RES de DISC, que sí interpreta.
- `BACK` → "Suele citar autores externos al estudio actual" — diferencia BACK de DISC (que habla del estudio actual, no de trabajos previos).
- `DISC` → "Interpreta los resultados del estudio actual" — el "actual" es clave para no confundir con BACK (que describe trabajos ajenos).
- `LIM` → "Puede incluir trabajo futuro" — en este esquema, trabajo futuro va dentro de LIM, no en una categoría separada.
- `CONTR` → "Identifica explícitamente los aportes" — énfasis en "explícitamente"; DISC también puede mencionar aportes pero de forma implícita.
- `CONC` → "Resume los principales hallazgos" — diferencia de RES (que presenta resultados sin resumirlos en el cierre del paper).

---

### "Responde ÚNICAMENTE con un JSON válido en el formato exacto: {...}"

**Por qué JSON y no texto libre**:

1. **Parseo robusto**: `json.loads()` es determinista. Con texto libre el parseo depende de regex frágiles que se rompen si el modelo cambia ligeramente su formato de respuesta.

2. **Sin chain-of-thought**: para clasificación con definiciones claras, pedirle al modelo que razone paso a paso no mejora consistentemente la calidad y multiplica el costo de salida (de ~40 tokens a ~200-400 por request). Las definiciones detalladas en el prompt reemplazan el chain-of-thought.

3. **`confidence` sin costo adicional**: al pedirlo dentro del mismo JSON, se obtiene la certeza implícita del modelo sin una llamada extra. Sirve para el análisis de errores: predicciones de baja confidence son candidatas a revisión manual.

4. **`max_output_tokens=60` es suficiente**: `{"label": "INTRO", "confidence": 0.95}` tiene ~15 tokens. Sin este límite el modelo puede generar párrafos de explicación aunque se lo hayas prohibido.

---

### "No incluyas explicaciones, markdown ni texto fuera del JSON."

**Por qué es necesario**: los LLMs tienden a agregar texto como "Claro, el fragmento pertenece a..." antes del JSON, o a envolverlo en triple-backtick (```json```). La instrucción explícita reduce ese comportamiento. No lo elimina completamente, por eso el parser también limpia markdown antes de parsear.

---

## El mensaje de usuario (zero-shot)

```
Clasifica el siguiente fragmento de un artículo científico en español:

<texto del fragmento, máximo 700 palabras>
```

**Simple por diseño**: todo el contexto de la tarea está en el `system_instruction`. El mensaje de usuario solo introduce el texto a clasificar. Para Gemini en zero-shot se puede pasar directamente como string a `generate_content()`:

```python
response = gemini_model.generate_content(
    "Clasifica el siguiente fragmento de un artículo científico en español:\n\nTexto aquí..."
)
```

---

## El prompt few-shot: cómo se construye en Gemini

Para few-shot el "prompt" es una lista de turns alternados (conversación multi-turn):

```
[user]   → "Clasifica: <ejemplo de INTRO truncado a 150 palabras>"
[model]  → '{"label": "INTRO", "confidence": 1.0}'
[user]   → "Clasifica: <ejemplo de BACK truncado a 150 palabras>"
[model]  → '{"label": "BACK", "confidence": 1.0}'
...
(6 pares más, uno por etiqueta restante)
...
[user]   → "Clasifica: <fragmento real a predecir>"
```

El system_instruction ya está en el modelo — no se repite aquí.

En código:

```python
contents = [
    {"role": "user",  "parts": ["Clasifica: <ejemplo INTRO>"]},
    {"role": "model", "parts": ['{"label": "INTRO", "confidence": 1.0}']},
    ...
    {"role": "user",  "parts": ["Clasifica: <fragmento real>"]},
]
response = gemini_model.generate_content(contents)
```

**Diferencia con OpenAI**: Gemini usa `"model"` como role (no `"assistant"`) y `"parts"` como campo del contenido (no `"content"`). Si se usa el formato de OpenAI con Gemini, la API devuelve error.

**Por qué `confidence: 1.0` en los ejemplos**: en los ejemplos se conoce la etiqueta correcta con certeza (son del TRAIN_TEST validado). Poner 1.0 refuerza el patrón de output esperado.

**Por qué k=1 y no k=2**: con 8 etiquetas, k=2 significa 16 pares de ejemplo ≈ +2,400 palabras por request. Sube el costo de tokens y en free tier puede hacer que requests largos tarden más o fallen. k=1 es el punto de equilibrio.

**Por qué truncar ejemplos a 150 palabras**: los ejemplos sirven para mostrar el formato de output y el estilo retórico de cada categoría, no para mostrar el fragmento completo. 150 palabras captura suficiente contexto para eso.

**Por qué el fragmento más cercano a la mediana**: evita ejemplos outlier en longitud. Si el ejemplo de CONC es de 950 palabras y la mayoría son de 400, el modelo puede asociar CONC con textos muy largos.

---

## Qué no tiene el prompt y por qué

**No tiene chain-of-thought**: las definiciones de las etiquetas ya son suficientemente precisas. CoT multiplicaría el costo de tokens de salida entre 5 y 10 veces sin mejora clara en clasificación multiclase con categorías bien definidas.

**No tiene instrucciones de desempate**: cuando un fragmento es genuinamente ambiguo (puede ser DISC o CONC), no se le dice al modelo cómo desempatar. Es intencional — se quiere medir la confusión natural del modelo, no suprimirla artificialmente.

**No tiene ejemplos negativos ("esto NO es INTRO")**: añaden complejidad al prompt sin mejora clara para clasificación multiclase con definiciones positivas bien especificadas.

---

## Comparación zero-shot vs few-shot: qué esperar

| Aspecto | Zero-shot | Few-shot k=1 |
|---|---|---|
| Tokens por request (aprox.) | ~600–800 | ~1,400–1,700 |
| Costo equiv. tier pago (relativo) | 1× | ~2× |
| Calidad esperada | Buena | Generalmente mejor |
| Mejora esperada | — | Etiquetas ambiguas (CONTR, LIM) |
| Riesgo | Modelo interpreta etiquetas a su manera | Ejemplos pobres pueden sesgar |
| Tiempo en free tier | ~2 horas | ~2 horas |

Si la diferencia entre zero-shot y few-shot es menor a 2 puntos de Macro F1, significa que las definiciones del system prompt son suficientemente informativas por sí solas. Si es grande, hay etiquetas que el modelo no puede discriminar solo con definiciones textuales.

---

## Resumen: las tres decisiones de diseño más importantes

1. **Definiciones del enunciado, no reformuladas**: garantiza que el modelo usa los mismos criterios que los anotadores humanos. Reformularlas introduciría divergencia entre las métricas y el ground truth.

2. **JSON forzado con `confidence`**: permite parseo robusto, controla el costo de salida con `max_output_tokens=60`, y captura la certeza del modelo sin costo adicional.

3. **`temperature=0`**: máximo determinismo disponible en Gemini (no tiene parámetro seed). Hace el experimento reproducible — mismos fragmentos producen siempre las mismas predicciones.
