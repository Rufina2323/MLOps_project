from clearml import Task
from sklearn.tree import DecisionTreeRegressor
import joblib

from data_utils import load_data, calculate_metrics


def run_experiment(max_depth, min_samples_split, min_samples_leaf, experiment_name):

    task = Task.create(
        project_name="MLOps Wine Quality",
        task_name=experiment_name
    )

    task.mark_started()

    params = {
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "random_state": 42
    }
    task.connect(params)

    X_train, X_test, y_train, y_test = load_data("../data/winequality-red.csv")

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

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
        {"max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1},
        {"max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1},
        {"max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
        {"max_depth": None, "min_samples_split": 10, "min_samples_leaf": 2},
    ]

    for i, exp in enumerate(experiments):
        run_experiment(
            max_depth=exp["max_depth"],
            min_samples_split=exp["min_samples_split"],
            min_samples_leaf=exp["min_samples_leaf"],
            experiment_name=f"tree_exp_{i}"
        )


if __name__ == "__main__":
    main()
