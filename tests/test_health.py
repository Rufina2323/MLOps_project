class TestHealth:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_model_loaded_flag(self, client):
        data = client.get("/health").json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_response_structure(self, client):
        data = client.get("/health").json()
        assert set(data.keys()) == {"status", "model_loaded"}
