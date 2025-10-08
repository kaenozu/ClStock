from fastapi.testclient import TestClient


def test_personal_dashboard_import_and_static_file_serving():
    from app.personal_dashboard import app

    client = TestClient(app)
    response = client.get("/static/style.css")
    assert response.status_code == 200
    content = response.json()
    assert isinstance(content, str)
    assert content.strip()
