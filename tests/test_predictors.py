def test_imports():
    from app.main import app
    assert app is not None

def test_health():
    from app.main import app
    client=app.test_client()
    r=client.get("/api/health")
    assert r.status_code==200
    assert r.json["ok"] is True
