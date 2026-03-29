import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
api_dir = project_root / "api"
sys.path.insert(0, str(api_dir))

os.environ.setdefault("PROJECT_ROOT", str(project_root))


@pytest.fixture(scope="session")
def app():
    from main import app
    return app


@pytest.fixture(scope="session")
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_wine_data():
    return {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }


@pytest.fixture()
def another_wine_data():
    return {
        "fixed_acidity": 11.2,
        "volatile_acidity": 0.28,
        "citric_acid": 0.56,
        "residual_sugar": 1.9,
        "chlorides": 0.075,
        "free_sulfur_dioxide": 17.0,
        "total_sulfur_dioxide": 60.0,
        "density": 0.998,
        "pH": 3.16,
        "sulphates": 0.58,
        "alcohol": 9.8,
    }
