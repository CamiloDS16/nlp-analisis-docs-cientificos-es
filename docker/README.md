# Despliegue del sistema — Docker Compose en EC2

Este directorio contiene los archivos para levantar el sistema completo: backend FastAPI (puerto 8000) + demo Streamlit (puerto 8501) como dos contenedores orquestados.

---

## Requisitos previos

- Instancia EC2: `t3.small` (2 vCPU, 2 GB RAM) con Ubuntu 24.04 LTS
- Security group con puertos **8000** y **8501** abiertos (inbound TCP, 0.0.0.0/0)
- Docker y Docker Compose instalados en la instancia
- Clave SSH para conectarse al EC2

---

## Estructura esperada en el servidor

```
nlp-docs-cientificos-es/          ← clon del repo (git clone)
├── api/
├── demo/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/
│   ├── task1_encoder/            ← pesos del encoder T1 (transferir via SCP)
│   └── task2_encoder/            ← pesos del encoder T2 (pendiente)
└── .env                          ← API keys (NO está en git — crear manualmente)
```

Los pesos de los modelos **no están en el repositorio** (excluidos en `.gitignore`). Se transfieren directamente al EC2 via SCP.

---

## Paso a paso para reproducir el despliegue

### 1. Clonar el repositorio en EC2

```bash
git clone https://github.com/<org>/nlp-docs-cientificos-es.git
cd nlp-docs-cientificos-es
```

### 2. Crear el archivo `.env` con las API keys

```bash
cat > .env << 'EOF'
GOOGLE_API_KEY=tu_clave_google_aqui
# Opcional si se usa Ollama local:
# OLLAMA_URL=http://localhost:11434
# OLLAMA_MODEL_T1=llama3
# OLLAMA_MODEL_T2=llama3
EOF
```

### 3. Transferir los pesos del encoder desde tu máquina local

Desde tu **máquina local** (no desde EC2), con la IP pública de la instancia:

```bash
# Encoder Task 1
scp -i tu_clave.pem -r models/task1_encoder/ ubuntu@<IP_EC2>:~/nlp-docs-cientificos-es/models/

# Encoder Task 2 (cuando esté disponible)
scp -i tu_clave.pem -r models/task2_encoder/ ubuntu@<IP_EC2>:~/nlp-docs-cientificos-es/models/
```

Verificar que los pesos quedaron en la ruta correcta:

```bash
ls models/task1_encoder/
# Debe mostrar: config.json  model.safetensors  tokenizer*  (o equivalente)
```

### 4. Limpiar espacio en disco antes de construir (si es primera vez)

PyTorch ocupa ~2 GB en la imagen. Si el disco está lleno de imágenes antiguas:

```bash
docker system prune -a -f
```

### 5. Construir y levantar los contenedores

```bash
cd docker
docker compose up --build -d
```

- `--build` reconstruye la imagen del API desde el `Dockerfile`
- `-d` corre en segundo plano (detached)

### 6. Verificar que ambos contenedores están corriendo

```bash
docker ps
```

Debe mostrar dos contenedores activos:

```
NAMES               PORTS
docker-api-1        0.0.0.0:8000->8000/tcp
docker-demo-1       0.0.0.0:8501->8501/tcp
```

### 7. Verificar salud del backend

```bash
curl http://localhost:8000/health
# Respuesta esperada: {"status": "ok"}
```

Desde un navegador externo (reemplazar con la IP pública actual):

```
http://<IP_EC2>:8000/health
http://<IP_EC2>:8501
```

---

## Actualizar el código sin reconstruir la imagen

Si solo cambiaste código Python (no `requirements.txt` ni `Dockerfile`):

```bash
git pull
docker compose down && docker compose up -d
```

Si cambiaste dependencias:

```bash
git pull
docker compose up --build -d
```

---

## Arquitectura de los contenedores

```
Usuario → http://<IP>:8501  →  demo (Streamlit, python:3.11-slim)
                                  ↓ HTTP POST /analyze
              http://api:8000  →  api (FastAPI/uvicorn, imagen custom)
                                  ├── Gemini 2.5 Flash (via GOOGLE_API_KEY)
                                  ├── Encoder T1 (models/task1_encoder/ montado como volumen)
                                  └── Encoder T2 (models/task2_encoder/ montado como volumen)
```

**Red interna Docker**: el frontend resuelve el backend como `http://api:8000` usando el DNS interno de Docker Compose. La URL se escribe a un archivo `.backend_url` al inicio del contenedor demo — Streamlit la lee en cada llamada porque cachea las variables de entorno al importar el módulo.

**Volumen de modelos**: los pesos no van dentro de la imagen. Se montan en runtime desde `../models:/app/models`. Así se pueden actualizar los pesos sin reconstruir la imagen.

---

## Endpoints del API

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Verificación de vida del servidor |
| GET | `/models/{task}` | Lista modelos disponibles para `task1` o `task2` |
| POST | `/analyze` | Analiza texto completo (T1 + T2 sobre cada párrafo) |
| POST | `/compare` | Fija segmentación T1 y compara los tres modelos T2 |

---

## Modelos disponibles

| Slot | Modelo T1 | Modelo T2 | Estado |
|------|-----------|-----------|--------|
| `commercial-api-gemini` | Gemini 2.5 Flash | Gemini 2.5 Flash | Activo |
| `encoder-scibeto` | SciBETO fine-tuned (Sergio) | Pendiente | T1 activo |
| `openweight` | LLaMA 3 via Ollama | LLaMA 3 via Ollama | Requiere t3.large |

---

## Nota sobre la IP pública

AWS Academy cambia la IP pública cada vez que se reinicia la instancia. Para obtener la IP actual:

```bash
# Desde la consola AWS, o desde dentro del EC2:
curl -s http://169.254.169.254/latest/meta-data/public-ipv4
```
