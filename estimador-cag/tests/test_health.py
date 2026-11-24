from app.config import get_settings


def test_health_reports_the_service_status(client):
    settings = get_settings()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": settings.app_env,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
    }


def test_health_does_not_expose_api_keys(client):
    settings = get_settings()

    body = client.get("/health").text

    assert settings.llm_api_key.get_secret_value() not in body


def test_swagger_docs_are_available(client):
    assert client.get("/docs").status_code == 200

    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "Estimador CAG"
    assert "/api/v1/estimate" in schema["paths"]
    assert "/health" in schema["paths"]
