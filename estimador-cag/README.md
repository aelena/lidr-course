# Estimador CAG

<!-- badges-start -->
[![Tests](https://img.shields.io/github/actions/workflow/status/aelena/lidr-course/tests.yml?style=flat&logo=github&label=tests)](https://github.com/aelena/lidr-course/actions/workflows/tests.yml) [![Version](https://img.shields.io/badge/version-0.1.0-blue?style=flat)](https://github.com/aelena/lidr-course/blob/estimador-cag/estimador-cag/pyproject.toml) [![Python](https://img.shields.io/badge/python-3.14+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/) [![FastAPI](https://img.shields.io/badge/fastapi-0.165+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) [![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
<!-- badges-end -->

API que genera estimaciones de proyectos de software a partir de la transcripción de una reunión con el cliente.

Usa **CAG (Cache-Augmented Generation)**: las estimaciones anteriores de la empresa viajan en el prompt de cada llamada al LLM, y el modelo las toma como referencia para calibrar tareas, horas, equipo y duración. No hay base de datos vectorial ni paso de recuperación.

## Características

- Endpoint `POST /api/v1/estimate` que recibe una transcripción y devuelve la estimación en markdown.
- Proveedor configurable: OpenAI o Anthropic, elegido con `LLM_PROVIDER`.
- Salida estructurada: el modelo devuelve las tareas con sus horas y el total se calcula en código, así que siempre coincide con el desglose.
- Ejemplos de contexto en JSON (`app/context/examples.json`), fáciles de ampliar sin tocar código.
- Configuración validada al arrancar con `pydantic-settings`.
- Documentación interactiva (Swagger) en `/docs`.

## Requisitos

- Python 3.14 o superior
- [uv](https://docs.astral.sh/uv/)
- Una API key de OpenAI o de Anthropic

## Instalación

```bash
git clone https://github.com/aelena/lidr-course.git
cd lidr-course/estimador-cag
uv sync
```

## Configuración

Copia la plantilla y rellena la clave del proveedor que vayas a usar:

```bash
cp .env.example .env
```

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `OPENAI_API_KEY` | API key de OpenAI | Sin valor; obligatoria si `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | API key de Anthropic | Sin valor; obligatoria si `LLM_PROVIDER=anthropic` |
| `LLM_PROVIDER` | Proveedor a utilizar: `openai` o `anthropic` | `openai` |
| `LLM_MODEL` | Modelo a utilizar | `gpt-4o-mini` |
| `APP_ENV` | Entorno de ejecución | `development` |
| `LOG_LEVEL` | Nivel de logging (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) | `DEBUG` |

La aplicación no arranca si falta la clave del proveedor seleccionado. Si cambias a Anthropic, cambia también `LLM_MODEL` (por ejemplo, `claude-haiku-4-5`).

`.env` está en `.gitignore`: las claves no se versionan.

## Uso

Arranca el servidor:

```bash
uv run uvicorn app.main:app --reload
```

La API queda en `http://localhost:8000` y la documentación en `http://localhost:8000/docs`.

### Generar una estimación

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{"transcription": "El cliente necesita una landing page con formulario de contacto y un blog. El plazo ideal es de 4 semanas."}'
```

Respuesta:

```json
{
  "estimation": "## Estimación: Landing Page y Blog

### Desglose de tareas:
1. ...",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "finish_reason": "stop",
  "total_hours": 130,
  "input_tokens": 1013,
  "output_tokens": 122,
  "validation": null
}
```

Solo `transcription` es obligatorio. El resto de campos del cuerpo ajustan cómo se construye el prompt y cómo se llama al modelo:

| Campo | Descripción | Valor por defecto |
|---|---|---|
| `preprocessing` | Normaliza espacios y líneas en blanco de la transcripción | `true` |
| `use_examples` | Inyecta las estimaciones anteriores; `false` permite comparar sin contexto | `true` |
| `num_examples` | Cuántas estimaciones anteriores inyectar | todas |
| `example_format` | Formato de los ejemplos en el prompt: `markdown` o `json` | `markdown` |
| `model` | Modelo a usar en lugar de `LLM_MODEL` | `LLM_MODEL` |
| `max_tokens` | Máximo de tokens que puede generar el modelo | el del proveedor |
| `thinking_budget` | Tokens de razonamiento para modelos de Anthropic que aceptan presupuesto (por ejemplo, `claude-haiku-4-5`). OpenAI lo ignora | sin razonamiento |
| `evaluate` | Añade a la respuesta, en `validation`, una comprobación de la estructura de la estimación | `false` |

Con `"evaluate": true`, `validation` indica si la estimación tiene título, desglose, total, equipo y duración, si el total coincide con la suma de las tareas y si el modelo terminó sin cortarse.

En Windows, Git Bash y PowerShell pueden enviar los acentos en una codificación que la API rechaza con un 400. Guarda el cuerpo en un archivo UTF-8 y envíalo con `--data-binary "@body.json"`, o usa Swagger.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/v1/estimate` | Genera una estimación a partir de una transcripción |
| `GET` | `/health` | Estado del servicio, entorno, proveedor y modelo |
| `GET` | `/docs` | Documentación Swagger |
| `GET` | `/redoc` | Documentación ReDoc |

Un cuerpo inválido (por ejemplo, una transcripción vacía) devuelve `422`. Un fallo del proveedor de LLM, o una respuesta cortada por `max_tokens`, devuelve `500` con el detalle.

## Estructura del proyecto

```
estimador-cag/
├── app/
│   ├── main.py              # Aplicación FastAPI y endpoint /health
│   ├── config.py            # Configuración desde .env y logging (structlog)
│   ├── routers/
│   │   └── estimations.py   # Endpoint HTTP
│   ├── schemas/
│   │   └── estimation.py    # Schemas de request y response
│   ├── services/
│   │   ├── llm_service.py   # Construcción del prompt y llamada al LLM
│   │   └── evaluation.py    # Validación de la estructura de una estimación
│   └── context/
│       ├── examples.json    # Estimaciones anteriores (datos)
│       └── examples.py      # Carga y renderizado de los ejemplos
├── tests/
├── .env.example
└── pyproject.toml
```

`routers/` gestiona HTTP, `schemas/` define los contratos de la API, `services/` contiene la lógica de negocio y `context/` guarda los datos estáticos que se inyectan en el prompt.

### Añadir ejemplos de contexto

Añade una entrada a `app/context/examples.json` con `meeting_summary`, `title`, `tasks` (nombre y horas), `team` y `duration`. El total de horas se calcula a partir de las tareas.

## Tests

```bash
uv run pytest
```

Los tests no llaman al LLM: sustituyen el servicio por un doble, así que no necesitan API key ni consumen tokens.

Hay además un test de evaluación que sí llama al proveedor configurado en `.env`. Solo se ejecuta si se pide:

```bash
RUN_LLM_EVAL=1 uv run pytest
```

## Contribuir

Las contribuciones son bienvenidas:

1. Abre un issue para comentar el cambio que propones.
2. Crea una rama a partir de `main`.
3. Añade tests para el comportamiento nuevo y comprueba que `uv run pytest` pasa.
4. Abre un pull request describiendo el cambio.

## Licencia

Este proyecto todavía no tiene una licencia definida. Hasta que se añada un archivo `LICENSE`, se aplican todos los derechos reservados.
