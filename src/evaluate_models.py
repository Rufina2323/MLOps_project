import os
import joblib
import pandas as pd

from clearml import Task
from data_utils import load_data, calculate_metrics


def load_models(models_dir):
    models = {}

    for file in os.listdir(models_dir):
        if file.endswith(".pkl"):
            model_name = file.replace(".pkl", "")
            model_path = os.path.join(models_dir, file)
            models[model_name] = joblib.load(model_path)

    return models


def main():
    task = Task.init(
        project_name="MLOps Wine Quality",
        task_name="Model Evaluation"
    )

    _, X_test, _, y_test = load_data("../data/winequality-red.csv")

    models_dir = "models/"
    models = load_models(models_dir)

    results = []

    logger = task.get_logger()

    for name, model in models.items():
        y_pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)

        results.append({
            "model": name,
            **metrics
        })

        for metric_name, value in metrics.items():
            logger.report_scalar(
                title=name,
                series=metric_name,
                value=value,
                iteration=0
            )

    results_df = pd.DataFrame(results)
    print("\nComparison of models:\n")
    print(results_df)

    results_df.to_csv("model_comparison.csv", index=False)

    task.upload_artifact("comparison_table", "model_comparison.csv")

    task.close()


if __name__ == "__main__":
    main()