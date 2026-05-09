# Despliegue — Docker Compose en EC2

Dos contenedores orquestados: FastAPI en el 8000, Streamlit en el 8501. La instancia es una `t3.small` (2 vCPU, 2 GB RAM) con Ubuntu 24.04 en AWS Academy. Los puertos 8000 y 8501 deben estar abiertos en el security group.

---

## Prerrequisitos

- Docker >= 24 y Docker Compose V2 (`docker compose`, sin guión) instalados en la instancia.
- Los pesos de los modelos **no están en el repositorio** (`.gitignore` excluye `*.safetensors`, `*.bin`, `*.pt`). Se transfieren al EC2 por SCP antes del primer despliegue.
- Un archivo `.env` en la raíz del repositorio con las API keys. No se sube a git.

---

## Primer despliegue

```bash
# En la instancia EC2
git clone <repo_url> && cd nlp-docs-cientificos-es

# Crear .env
echo "GOOGLE_API_KEY=<clave>" > .env
```

Transferir los pesos desde la máquina local antes de levantar los contenedores:

```bash
# Desde la máquina local
scp -i <clave.pem> -r models/task1_encoder/ ubuntu@<IP>:~/nlp-docs-cientificos-es/models/
```

Los pesos se montan como volumen (`../models:/app/models`) para no requerir reconstruir la imagen cada vez que cambia un modelo.

```bash
cd docker
docker compose up --build -d
```

Verificar que los dos contenedores están corriendo y el backend responde:

```bash
docker compose ps
curl -s localhost:8000/health
```

---

## Actualizar código

Si el cambio es solo en archivos Python (sin tocar `requirements.txt` ni `Dockerfile`):

```bash
git pull && docker compose down && docker compose up -d
```

Si cambiaron dependencias:

```bash
git pull && docker compose up --build -d
```

Para ver logs en tiempo real:

```bash
docker compose logs -f api
docker compose logs -f demo
```

Si el disco está lleno por capas Docker antiguas (sucede con t3.small y PyTorch):

```bash
docker system prune -a -f
```

---

## Arquitectura

```
Puerto 8501  →  demo  (python:3.11-slim, sin imagen custom)
                  │  POST /analyze, /compare
              http://api:8000
Puerto 8000  →  api   (imagen desde docker/Dockerfile)
                  ├── Gemini 2.5 Flash  — via GOOGLE_API_KEY en .env
                  ├── Encoder T1        — models/task1_encoder/ (volumen)
                  └── Encoder T2        — models/task2_encoder/ (volumen)
```

El frontend resuelve el backend por DNS interno de Docker Compose como `http://api:8000`. La URL se escribe a un archivo `.backend_url` al iniciar el contenedor demo en lugar de pasarla como variable de entorno, porque Streamlit resuelve las variables de entorno al importar el módulo, no en cada request.

---

## Endpoints

| Ruta | Descripción |
|------|-------------|
| `GET /health` | Liveness check |
| `GET /models/{task}` | Modelos disponibles para `task1` o `task2` |
| `POST /analyze` | Segmenta el texto en párrafos y corre T1 + T2 sobre cada uno |
| `POST /compare` | Fija la segmentación T1 y corre los tres modelos T2 en paralelo |

---

## Estado actual de los modelos

| Slot | T1 | T2 | Estado |
|------|----|----|--------|
| `commercial-api-gemini` | Gemini 2.5 Flash, few-shot k=3 (Macro F1=0.497) | Gemini 2.5 Flash, zero-shot | Activo |
| `encoder-scibeto` | SciBETO fine-tuned (Sergio) en `models/task1_encoder/` | Pendiente | T1 activo, T2 sin artefacto |
| `openweight` | LLaMA 3 vía Ollama | LLaMA 3 vía Ollama | Requiere t3.large — t3.small no tiene RAM para el modelo 8B |

---

## Nota sobre la IP

AWS Academy asigna una IP pública nueva en cada arranque de la instancia. La IP actual se obtiene desde dentro del EC2 con:

```bash
curl -s http://169.254.169.254/latest/meta-data/public-ipv4
```
