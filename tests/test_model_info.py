class TestModelInfo:

    def test_model_info_returns_200(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_has_name(self, client):
        data = client.get("/model-info").json()
        assert "model_name" in data
        assert isinstance(data["model_name"], str)

    def test_model_info_has_params(self, client):
        data = client.get("/model-info").json()
        assert "model_params" in data
        assert isinstance(data["model_params"], dict)

    def test_model_info_has_metrics(self, client):
        data = client.get("/model-info").json()
        assert "metrics" in data

    def test_model_info_ridge_params(self, client):
        data = client.get("/model-info").json()
        params = data["model_params"]
        assert "alpha" in params
        assert "fit_intercept" in params

    def test_model_info_has_training_date(self, client):
        data = client.get("/model-info").json()
        assert "training_date" in data

    def test_model_info_has_feature_columns(self, client):
        data = client.get("/model-info").json()
        if data.get("feature_columns"):
            assert len(data["feature_columns"]) == 11
