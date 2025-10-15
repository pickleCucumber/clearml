import warnings
from typing import Optional

from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


# создаем датасет
@PipelineDecorator.component(
    cache=True,
    task_type=TaskTypes.data_processing,
)
def create_dataset(project: str, dataset_name: str, date: str, date_start: str) -> str:
    from clearml import Dataset, Task

    from src.constants import RAW_DATA_DIR
    from src.data.data_load import load_dataset

    task = Task.current_task()
    logger = task.get_logger()

    logger.report_text("Starting create_dataset process...")

    load_dataset(date=date, date_start=date_start)
    logger.report_text("Dataset loaded successfully.")

    logger.report_text("Creating new dataset in ClearML...")
    dataset = Dataset.create(dataset_project=project, dataset_name=dataset_name)
    dataset.add_files(RAW_DATA_DIR)
    dataset.upload()
    dataset.finalize()
    logger.report_text(f"Dataset {dataset.id} created successfully.")

    return dataset.id


if __name__ == "__main__":
    PipelineDecorator.run_locally()

    create_dataset("nonres", "test")
