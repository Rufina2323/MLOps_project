from clearml import Task
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from data_utils import load_data, calculate_metrics


def run_experiment(alpha, fit_intercept, experiment_name):

    task = Task.create(
        project_name="MLOps Wine Quality",
        task_name=experiment_name
    )

    task.mark_started()

    params = {
        "alpha": alpha,
        "fit_intercept": fit_intercept,
        "random_state": 42
    }
    task.connect(params)

    X_train, X_test, y_train, y_test = load_data("../data/winequality-red.csv")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)

    logger = task.get_logger()
    for name, value in metrics.items():
        logger.report_scalar("metrics", name, value, iteration=0)

    model_path = f"models/model_{experiment_name}.pkl"
    joblib.dump(model, model_path)
    task.upload_artifact("model", model_path)
    task.mark_completed()
    task.close()


def main():
    experiments = [
        {"alpha": 0.1, "fit_intercept": True},
        {"alpha": 1.0, "fit_intercept": True},
        {"alpha": 10.0, "fit_intercept": True},
        {"alpha": 1.0, "fit_intercept": False},
    ]

    for i, exp in enumerate(experiments):
        run_experiment(
            alpha=exp["alpha"],
            fit_intercept=exp["fit_intercept"],
            experiment_name=f"ridge_exp_{i}"
        )


if __name__ == "__main__":
    main()
