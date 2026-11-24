# Estimador CAG

<!-- badges-start -->
[![Tests](https://img.shields.io/github/actions/workflow/status/aelena/lidr-course/tests.yml?style=flat&logo=github&label=tests)](https://github.com/aelena/lidr-course/actions/workflows/tests.yml) [![Version](https://img.shields.io/badge/version-0.1.0-blue?style=flat)](https://github.com/aelena/lidr-course/blob/estimador-cag/estimador-cag/pyproject.toml) [![Python](https://img.shields.io/badge/python-3.14+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/) [![FastAPI](https://img.shields.io/badge/fastapi-0.165+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) [![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
<!-- badges-end -->

API que genera estimaciones de proyectos de software a partir de la transcripción de una reunión con el cliente.

Usa **CAG (Context-Augmented Generation)**: las estimaciones anteriores de la empresa viajan en el prompt de cada llamada al LLM, y el modelo las toma como referencia para calibrar tareas, horas, equipo y duración. No hay base de datos vectorial ni paso de recuperación.

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
  "estimation": "## Estimación: Landing Page y Blog\n\n### Desglose de tareas:\n1. ...",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "total_hours": 130,
  "input_tokens": 1013,
  "output_tokens": 122
}
```

En Windows, Git Bash y PowerShell pueden enviar los acentos en una codificación que la API rechaza con un 400. Guarda el cuerpo en un archivo UTF-8 y envíalo con `--data-binary "@body.json"`, o usa Swagger.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/v1/estimate` | Genera una estimación a partir de una transcripción |
| `GET` | `/health` | Estado del servicio, entorno, proveedor y modelo |
| `GET` | `/docs` | Documentación Swagger |

Una transcripción vacía devuelve `422`. Un fallo del proveedor de LLM devuelve `502`.

## Estructura del proyecto

```
estimador-cag/
├── app/
│   ├── main.py              # Aplicación FastAPI y endpoint /health
│   ├── config.py            # Configuración cargada desde .env
│   ├── routers/
│   │   └── estimations.py   # Endpoint HTTP y schemas de request/response
│   ├── services/
│   │   └── llm_service.py   # Construcción del prompt y llamada al LLM
│   └── context/
│       ├── examples.json    # Estimaciones anteriores (datos)
│       └── examples.py      # Carga y renderizado de los ejemplos
├── tests/
├── .env.example
└── pyproject.toml
```

`routers/` gestiona HTTP, `services/` contiene la lógica de negocio y `context/` guarda los datos estáticos que se inyectan en el prompt.

### Añadir ejemplos de contexto

Añade una entrada a `app/context/examples.json` con `meeting_summary`, `title`, `tasks` (nombre y horas), `team` y `duration`. El total de horas se calcula a partir de las tareas.

## Tests

```bash
uv run pytest
```

Los tests no llaman al LLM: sustituyen el servicio por un doble, así que no necesitan API key ni consumen tokens.

## Contribuir

Las contribuciones son bienvenidas:

1. Abre un issue para comentar el cambio que propones.
2. Crea una rama a partir de `main`.
3. Añade tests para el comportamiento nuevo y comprueba que `uv run pytest` pasa.
4. Abre un pull request describiendo el cambio.

## Licencia

Este proyecto todavía no tiene una licencia definida. Hasta que se añada un archivo `LICENSE`, se aplican todos los derechos reservados.
