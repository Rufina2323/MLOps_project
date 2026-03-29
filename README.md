# Model Selection Report

## Описание задачи

В рамках проекта была решена задача регрессии: предсказание оценки качества вина (`quality`) на основе набора физико-химических признаков.

## Проведённые эксперименты

В ходе работы были обучены и сравнены следующие модели:

- **Ridge Regression** (линейная модель с L2-регуляризацией)
- **Decision Tree Regressor**

Для каждой модели проводились эксперименты с различными гиперпараметрами.  
Все эксперименты отслеживались с помощью ClearML: логировались параметры моделей, метрики качества и сами модели как артефакты.

### Гиперпараметры экспериментов

**Decision Tree:**
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

Использованные значения:

- `{"max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1}`
- `{"max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1}`
- `{"max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2}`
- `{"max_depth": None, "min_samples_split": 10, "min_samples_leaf": 2}`

**Ridge Regression:**
- `alpha` — коэффициент регуляризации
- `fit_intercept` — добавление свободного члена

Использованные значения:

- `{"alpha": 0.1, "fit_intercept": True}`
- `{"alpha": 1.0, "fit_intercept": True}`
- `{"alpha": 10.0, "fit_intercept": True}`
- `{"alpha": 1.0, "fit_intercept": False}`

## Используемые метрики

Для оценки качества моделей использовались следующие метрики:

- **RMSE (Root Mean Squared Error)** — основная метрика качества  
- **MAE (Mean Absolute Error)** — средняя абсолютная ошибка  
- **R² (коэффициент детерминации)** — показывает, насколько хорошо модель объясняет дисперсию данных  

## Результаты экспериментов

| Model | RMSE | MAE | R² | Link |
|------|------|-----|-----|-----|
| model_tree_exp_0 | 0.465192 | 0.540976 | 0.288160 | [Open in ClearML](https://app.clear.ml/projects/e0ba2a2b587045228fc85099eb166bde/experiments/3fe9a0f32f6f420c81de44716a47d93c/output/execution) |
| model_tree_exp_1 | 0.433510 | 0.496386 | 0.336640 | [Open in ClearML](https://app.clear.ml/projects/e0ba2a2b587045228fc85099eb166bde/experiments/82ac6d0e394a43cc9426764e3bd6dcb8/output/execution) |
| model_tree_exp_2 | 0.549136 | 0.501889 | 0.159708 | [Open in ClearML](https://app.clear.ml/projects/e0ba2a2b587045228fc85099eb166bde/experiments/6ae7e317035a43c7a432b7a9b79a6ee6/output/execution) |
| model_tree_exp_3 | 0.573569 | 0.517272 | 0.122320 | [Open in ClearML](https://app.clear.ml/projects/e0ba2a2b587045228fc85099eb166bde/experiments/6ad6a3b5660a41939926bbd31978e1eb/output/execution) |
| model_ridge_exp_0 | 0.390026 | 0.503533 | 0.403178 | [Open in ClearML](https://app.clear.ml/projects/e0ba2a2b587045228fc85099eb166bde/experiments/6efc1183f60b4806b6abd2df0294e7bf/output/execution) |
| model_ridge_exp_1 | 0.390038 | 0.503560 | 0.403161 | [Open in ClearML](https://app.clear.ml/projects/e0ba2a2b587045228fc85099eb166bde/experiments/99250375e29b461aad8d09b2f521b4cc/output/execution) |
| model_ridge_exp_2 | 0.390177 | 0.503822 | 0.402948 | [Open in ClearML](https://app.clear.ml/projects/e0ba2a2b587045228fc85099eb166bde/experiments/bed5e25260e442628d79defc380a7a1a/output/execution) |
| model_ridge_exp_3 | 32.389686 | 5.656919 | -48.562962 | [Open in ClearML](https://app.clear.ml/projects/e0ba2a2b587045228fc85099eb166bde/experiments/25b82dcbf13b45a588492ae2326885df/output/execution) |

[Open model evaluation in ClearML](https://app.clear.ml/projects/e0ba2a2b587045228fc85099eb166bde/experiments/a416ea9dac7c451bbc2bc18e4c2ad8fd/output/execution)


## Выбор лучшей модели

Наилучшие результаты показала модель **Ridge Regression**.

Причины выбора:
- минимальное значение RMSE среди всех корректных экспериментов (~0.39)
- более высокое значение R² (~0.403) по сравнению с Decision Tree
- стабильные результаты между различными гиперпараметрами
- лучшее обобщение на тестовых данных

Decision Tree показала худшие результаты:
- более высокий RMSE
- более низкий R²
- признаки переобучения

## Итог

- Лучшая модель: **Ridge Regression**
- Основная метрика выбора: **RMSE**
- Используемая конфигурация: Ridge Regression с параметрами, близкими к:
  - `alpha = 0.1 `  
  - `fit_intercept = True`
