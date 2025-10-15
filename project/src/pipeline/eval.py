from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator


# опишем этап валидации модели catboost
@PipelineDecorator.component(
    cache=False,
    task_type=TaskTypes.qc,
    # helper_functions=[all_classification_metrics]
)
def eval_model(model, X_train, y_train, X_oot, y_oot, X_monthly, y_monthly):
    from clearml import Task

    from utils.metrics.evaluation import (
        evaluate_classification_performance,
        evaluate_monthly_data,
        preprocess_and_compute_shap_values,
    )

    cols_dict = {
        "S_hare_active_credit": "S_hare_active_credit_eq",
        "Score": "MailRu_Score",
        "MatchingLevel": "MailRu_MatchingLevel",
        "INTEGRALSCOREValueId": "MGFN_INTEGRALSCOREValueId",
        "LIFETIMEBINValueId": "MGFN_LIFETIMEBINValueId",
    }

    task = Task.current_task()
    logger = task.get_logger()

    preprocess = model[0]
    estimator = model[1]

    # Evaluate and compute SHAP for training data
    evaluate_classification_performance(model, X_train, y_train, "Training Data")
    preprocess_and_compute_shap_values(
        estimator, X_train, preprocess, cols_dict, "Training Data", True
    )

    # Evaluate and compute SHAP for OOT data
    evaluate_classification_performance(model, X_oot, y_oot, "OOT Data")
    preprocess_and_compute_shap_values(
        estimator, X_oot, preprocess, cols_dict, "OOT Data", True
    )

    # Evaluate monthly data (now also computes SHAP values)
    evaluate_monthly_data(
        model,
        X_monthly,
        y_monthly,
        estimator,
        preprocess,
        cols_dict=cols_dict,
        calibrated=True,
    )
