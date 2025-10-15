from clearml.automation.controller import PipelineDecorator

from src.pipeline.data import create_dataset
from src.pipeline.eval import eval_model
from src.pipeline.prep import preproc_dataset
from src.pipeline.train import train_lgbm, train_rf


@PipelineDecorator.pipeline(
    name="Credit-Line Нерезы",
    project="Credit-Line Нерезы",
    version="0.0.1",
    output_uri=True,
)
def run_pipeline(date=None, date_start="2021-08-11", lag=3):
    from datetime import datetime

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # TODO: add date_start=None if None get 2years from date

    dataset_id = create_dataset(
        project="Credit-Line Нерезы",
        dataset_name="Credit-Line Нерезы",
        date=date,
        date_start=date_start,
    )
    X_train, y_train, X_oot, y_oot, X_monthly, y_monthly = preproc_dataset(
        dataset_id, lag
    )

    model_lgbm = train_lgbm(X_train, y_train)
    model_rf = train_rf(X_train, y_train)

    eval_model(model_lgbm, X_train, y_train, X_oot, y_oot, X_monthly, y_monthly)
    eval_model(model_rf, X_train, y_train, X_oot, y_oot, X_monthly, y_monthly)


if __name__ == "__main__":
    PipelineDecorator.run_locally()

    run_pipeline()
