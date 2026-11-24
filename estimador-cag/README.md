# Estimador CAG

<!-- badges-start -->
[![Tests](https://img.shields.io/github/actions/workflow/status/aelena/lidr-course/tests.yml?style=flat&logo=github&label=tests)](https://github.com/aelena/lidr-course/actions/workflows/tests.yml) [![Version](https://img.shields.io/badge/version-0.1.0-blue?style=flat)](https://github.com/aelena/lidr-course/blob/estimador-cag/estimador-cag/pyproject.toml) [![Python](https://img.shields.io/badge/python-3.14+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/) [![FastAPI](https://img.shields.io/badge/fastapi-0.165+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) [![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
<!-- badges-end -->

API que genera estimaciones de proyectos de software a partir de la transcripción de una reunión con el cliente.

Usa **CAG (Cache-Augmented Generation)**: las estimaciones anteriores de la empresa viajan en el prompt de cada llamada al LLM, y el modelo las toma como referencia para calibrar tareas, horas, equipo y duración. No hay base de datos vectorial ni paso de recuperación.

## Características

- Endpoint `POST /api/v1/estimate` que recibe una transcripción y devuelve la estimación en markdown.
- Proveedor configurable: OpenAI o Anthropic, elegido con `LLM_PROVIDER`.
- Tres formas de tratar una transcripción con ruido: enviarla tal cual, dar instrucciones de limpieza en el prompt, o extraer antes los requisitos con una primera llamada.
- Validación opcional de cada estimación: estructura, sumas de horas y costes, y respuestas cortadas.
- Ejemplos de contexto en JSON (`app/context/examples.json`), fáciles de ampliar sin tocar código.
- Configuración validada al arrancar con `pydantic-settings`.
- Documentación interactiva (Swagger) en `/docs`.

## Arquitectura CAG

El "cache" de CAG es el mensaje de sistema: instrucciones y ejemplos definidos en código, que se ensamblan igual en cada llamada.

```mermaid
flowchart TB
    subgraph FASE1["FASE 1 · Construcción del cache — estático, definido en código"]
        direction TB
        I["<b>① Instrucciones del sistema</b><br/><i>app/services/llm_service.py</i><br/>· Contrato de salida<br/>· Reglas de pricing (62,50 / 50 €/h)<br/>· Persona y tono<br/>build_system_prompt()"]
        E["<b>② Ejemplos canónicos</b><br/><i>app/context/examples.py</i><br/>· 5 estimaciones modelo<br/>· Few-shot in-context learning<br/>· Inyectados literalmente<br/>RAW_EXAMPLES"]
        A["<b>③ Prompt assembler</b><br/>build_system_prompt<br/>format_examples_for_prompt"]
        C["<b>④ Mensaje de sistema ensamblado · EL CACHE</b><br/>instrucciones + hasta 5 ejemplos (3 por defecto)<br/><i>Idéntico llamada tras llamada: es el prefijo<br/>que los proveedores pueden cachear</i>"]
        I --> A
        E --> A
        A --> C
    end
    subgraph FASE2["FASE 2 · Inferencia — dinámico, una vez por petición"]
        direction LR
        U["<b>⑤ Petición del usuario</b><br/><i>app/schemas/estimation.py</i><br/>transcription<br/>preprocessing<br/>EstimationRequest<br/><i>solo esto cambia entre llamadas</i>"]
        M["<b>⑥ _complete_with_openai() · _complete_with_anthropic()</b><br/>messages = [<br/>role: system → content: ⟨ cache ⟩<br/>role: user → content: ⟨ transcription ⟩<br/>]"]
        L["<b>⑦ Llamada al modelo</b><br/><i>LLM provider</i><br/>chat.completions<br/>messages.create"]
        R["<b>⑧ Respuesta estructurada</b><br/>EstimationResponse"]
        U --> M
        M --> L
        L --> R
    end
    C -. "reutilizado en cada llamada" .-> M

    classDef instrucciones stroke:#6366f1,stroke-width:3px
    classDef ejemplos stroke:#10b981,stroke-width:3px
    classDef assembler stroke:#a855f7,stroke-width:3px
    classDef cache stroke:#22d3ee,stroke-width:3px
    classDef peticion stroke:#f59e0b,stroke-width:3px
    classDef respuesta stroke:#f43f5e,stroke-width:3px
    class I instrucciones
    class E ejemplos
    class A assembler
    class C,M cache
    class U peticion
    class L,R respuesta
```

El mensaje de sistema solo cambia si la petición cambia las opciones que lo construyen (`use_examples`, `num_examples`, `example_format`, `preprocessing`).

Que el proveedor cobre menos por él depende de su tamaño. Medido con `gpt-4o-mini` en llamadas repetidas:

| `num_examples` | Tokens del mensaje de sistema | Tokens servidos desde cache |
|---|---|---|
| `3` (por defecto) | 895 | 0 |
| `5` | 1473 | 1408 |

Con los tres ejemplos por defecto el mensaje queda por debajo del mínimo a partir del cual OpenAI cachea la entrada; con los cinco lo supera y casi todo el prefijo se sirve desde cache a partir de la segunda llamada. Con Anthropic habría que marcar además el bloque con `cache_control`, cosa que el servicio no hace.

### Cómo viaja una petición

Ciclo de vida de `POST /api/v1/estimate`:

```mermaid
flowchart TB
    CLIENT["<b>CLIENT</b><br/>curl -X POST /api/v1/estimate<br/>{ transcription: …, preprocessing: none }"]

    subgraph API["API LAYER"]
        direction LR
        ROUTER["<i>app/routers/estimations.py</i><br/><b>create_estimation(request)</b><br/><i>orquesta: llamada al servicio · validación · build de la respuesta</i>"]
        SCHEMAS["<i>app/schemas/estimation.py</i><br/>EstimationRequest<br/>EstimationResponse<br/>EstimationValidation"]
    end

    subgraph SERVICE["SERVICE LAYER"]
        LLM["<i>app/services/llm_service.py</i><br/><b>generate_estimation() · build_system_prompt()</b><br/>_complete_with_openai() · _complete_with_anthropic()<br/><i>build del prompt · llamada al proveedor · unpack de la respuesta (text + usage + finish_reason)</i>"]
    end

    subgraph CONTEXT["CONTEXT"]
        EXAMPLES["<i>app/context/examples.py</i><br/>RAW_EXAMPLES (× 5)<br/>select_examples()<br/>format_examples_for_prompt()<br/><i>→ inyectados en el system prompt (CAG)</i>"]
    end

    subgraph VALIDATION["VALIDATION"]
        EVAL["<i>app/services/evaluation.py</i><br/><b>evaluate_estimation_structure(estimation, finish_reason)</b><br/><i>→ { is_valid, task_count, checks, issues }</i><br/>6 checks deterministas, 8 si hay costes · no llama al LLM"]
    end

    EXTERNAL["<b>EXTERNAL · OpenAI / Anthropic</b><br/>chat.completions<br/>messages<br/><i>coste · latencia</i>"]

    CLIENT --> ROUTER
    ROUTER -. valida con .- SCHEMAS
    ROUTER --> LLM
    EXAMPLES --> LLM
    LLM --> EXTERNAL
    ROUTER -- "si evaluate" --> EVAL

    classDef api stroke:#3b82f6,stroke-width:3px
    classDef service stroke:#a855f7,stroke-width:3px
    classDef context stroke:#10b981,stroke-width:3px
    classDef validation stroke:#f59e0b,stroke-width:3px
    classDef external stroke:#ef4444,stroke-width:3px
    class ROUTER,SCHEMAS api
    class LLM service
    class EXAMPLES context
    class EVAL validation
    class EXTERNAL external
```

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
  "estimation": "## Estimación: Landing Page y Blog ...",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "finish_reason": "stop",
  "total_hours": 130.0,
  "input_tokens": 1013,
  "output_tokens": 122,
  "latency_ms": 3850,
  "extracted_requirements": null,
  "validation": null
}
```

`estimation` es el markdown tal como lo escribe el modelo. `total_hours` no sale del modelo: se suma en código a partir de las tareas del desglose, y es `null` si no se pudo leer ninguna.

Solo `transcription` es obligatorio. El resto de campos del cuerpo ajustan cómo se construye el prompt y cómo se llama al modelo:

| Campo | Descripción | Valor por defecto |
|---|---|---|
| `preprocessing` | Tratamiento de una transcripción con ruido: `none`, `inline_cleaning` o `two_phase` | `none` |
| `use_examples` | Inyecta las estimaciones anteriores; `false` permite comparar sin contexto | `true` |
| `num_examples` | Máximo de estimaciones anteriores a inyectar | `3` |
| `example_format` | Formato de los ejemplos en el prompt: `markdown` o `json` | `markdown` |
| `model` | Modelo a usar en lugar de `LLM_MODEL` | `LLM_MODEL` |
| `max_tokens` | Máximo de tokens que puede generar el modelo | `4096` |
| `thinking_budget` | Tokens de razonamiento para modelos de Anthropic que aceptan presupuesto (por ejemplo, `claude-haiku-4-5`). OpenAI lo ignora | sin razonamiento |
| `evaluate` | Añade a la respuesta, en `validation`, una comprobación de la estructura de la estimación | `false` |

Modos de `preprocessing`:

- `none`: la transcripción se envía tal cual.
- `inline_cleaning`: el prompt de sistema indica al modelo cómo leer una reunión real (ignorar charla informal, resolver contradicciones, interpretar jerga).
- `two_phase`: una primera llamada al LLM convierte la transcripción en una lista de requisitos, que es lo que recibe la llamada de estimación. La lista vuelve en `extracted_requirements`, y los tokens de ambas llamadas se suman.

Con `"evaluate": true`, `validation` indica si la estimación tiene desglose de tareas, total de horas, equipo y duración, si el total que escribió el modelo coincide con la suma de las tareas (y lo mismo con los costes, cuando los hay) y si el modelo terminó sin cortarse. El modelo se equivoca al sumar con cierta frecuencia, y esta comprobación lo detecta.

### Formato de salida

`app/services/llm_service.py` tiene dos instrucciones de salida y una asignación que elige la activa:

- `PROMPT_OUTPUT_BASIC` (activa): pide una estimación sin fijar la estructura, así que el modelo imita la de los ejemplos.
- `PROMPT_OUTPUT_STRUCTURED`: fija resumen, tabla de tareas con horas y coste en EUR, totales, y riesgos y supuestos.

Para cambiarla, edita `ACTIVE_OUTPUT_PROMPT`. El prompt de sistema incluye las tarifas de referencia: 62,50 EUR/hora para desarrollo y 50 EUR/hora para diseño.

En Windows, Git Bash y PowerShell pueden enviar los acentos en una codificación que la API rechaza con un 400. Guarda el cuerpo en un archivo UTF-8 y envíalo con `--data-binary "@body.json"`, o usa Swagger.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/v1/estimate` | Genera una estimación a partir de una transcripción |
| `GET` | `/health` | Estado del servicio, entorno, proveedor y modelo |
| `GET` | `/docs` | Documentación Swagger |
| `GET` | `/redoc` | Documentación ReDoc |

Un cuerpo inválido (por ejemplo, una transcripción vacía) devuelve `422`. Un fallo del proveedor de LLM devuelve `500` con el detalle. Una respuesta cortada por `max_tokens` devuelve `200` con `finish_reason` distinto de `stop`.

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
