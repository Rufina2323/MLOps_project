class TestPredict:

    def test_predict_returns_200(self, client, sample_wine_data):
        response = client.post("/predict", json=sample_wine_data)
        assert response.status_code == 200

    def test_predict_returns_float(self, client, sample_wine_data):
        data = client.post("/predict", json=sample_wine_data).json()
        assert "prediction" in data
        assert isinstance(data["prediction"], float)

    def test_predict_returns_model_name(self, client, sample_wine_data):
        data = client.post("/predict", json=sample_wine_data).json()
        assert "model_name" in data
        assert isinstance(data["model_name"], str)
        assert len(data["model_name"]) > 0

    def test_predict_different_data(self, client, sample_wine_data, another_wine_data):
        pred1 = client.post("/predict", json=sample_wine_data).json()["prediction"]
        pred2 = client.post("/predict", json=another_wine_data).json()["prediction"]
        assert isinstance(pred1, float)
        assert isinstance(pred2, float)

    def test_predict_deterministic(self, client, sample_wine_data):
        pred1 = client.post("/predict", json=sample_wine_data).json()["prediction"]
        pred2 = client.post("/predict", json=sample_wine_data).json()["prediction"]
        assert pred1 == pred2
