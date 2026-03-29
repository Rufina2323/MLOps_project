import pytest


class TestValidationMissingFields:

    def test_empty_body(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_missing_single_field(self, client, sample_wine_data):
        data = sample_wine_data.copy()
        del data["alcohol"]
        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_missing_multiple_fields(self, client):
        response = client.post("/predict", json={"fixed_acidity": 7.4})
        assert response.status_code == 422

    def test_no_json_body(self, client):
        response = client.post("/predict")
        assert response.status_code == 422


class TestValidationInvalidTypes:

    def test_string_instead_of_float(self, client, sample_wine_data):
        data = sample_wine_data.copy()
        data["fixed_acidity"] = "not_a_number"
        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_null_value(self, client, sample_wine_data):
        data = sample_wine_data.copy()
        data["alcohol"] = None
        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_boolean_value(self, client, sample_wine_data):
        data = sample_wine_data.copy()
        data["density"] = True
        response = client.post("/predict", json=data)
        assert response.status_code in (200, 422)


class TestValidationOutOfRange:

    def test_negative_acidity(self, client, sample_wine_data):
        data = sample_wine_data.copy()
        data["fixed_acidity"] = -5.0
        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_alcohol_too_low(self, client, sample_wine_data):
        data = sample_wine_data.copy()
        data["alcohol"] = 1.0  # min 5
        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_alcohol_too_high(self, client, sample_wine_data):
        data = sample_wine_data.copy()
        data["alcohol"] = 99.0  # max 20
        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_ph_out_of_range(self, client, sample_wine_data):
        data = sample_wine_data.copy()
        data["pH"] = 0.5  # min 2.0
        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_density_out_of_range(self, client, sample_wine_data):
        data = sample_wine_data.copy()
        data["density"] = 5.0  # max 1.1
        response = client.post("/predict", json=data)
        assert response.status_code == 422


class TestValidationEdgeCases:

    def test_extra_fields_ignored(self, client, sample_wine_data):
        data = sample_wine_data.copy()
        data["extra_field"] = 999
        response = client.post("/predict", json=data)
        assert response.status_code == 200

    def test_integer_instead_of_float(self, client, sample_wine_data):
        data = sample_wine_data.copy()
        data["fixed_acidity"] = 7
        response = client.post("/predict", json=data)
        assert response.status_code == 200

    @pytest.mark.parametrize("field", [
        "fixed_acidity", "volatile_acidity", "citric_acid",
        "residual_sugar", "chlorides", "free_sulfur_dioxide",
        "total_sulfur_dioxide", "density", "sulphates", "alcohol",
    ])
    def test_each_field_required(self, client, sample_wine_data, field):
        data = sample_wine_data.copy()
        del data[field]
        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_ph_field_alias(self, client, sample_wine_data):
        data = sample_wine_data.copy()
        assert "pH" in data
        response = client.post("/predict", json=data)
        assert response.status_code == 200
