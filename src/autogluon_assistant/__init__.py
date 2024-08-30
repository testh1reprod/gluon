import datetime
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from hydra import compose, initialize
from omegaconf import OmegaConf
from rich import print as rprint
from typing_extensions import Annotated

from .assistant import TabularPredictionAssistant
from .task import TabularPredictionTask

logging.basicConfig(level=logging.INFO)

__all__ = ["TabularPredictionAssistant", "TabularPredictionTask"]


def _resolve_config_path(path: str):
    print(Path.cwd())
    return os.path.relpath(Path(path), Path(__file__).parent.absolute())


def get_task(path: Path) -> TabularPredictionTask:
    """Get a task from a path."""

    return TabularPredictionTask.from_path(path)


def make_prediction_outputs(task: TabularPredictionTask, predictions: pd.DataFrame) -> pd.DataFrame:
    test_ids = task.test_data[task.test_id_column]
    output_ids = task.output_data[task.output_id_column]

    if not test_ids.equals(output_ids):
        rprint("[orange]Warning: Test IDs and output IDs do not match![/orange]")

    outputs = pd.concat(
        [
            task.test_data[task.test_id_column],
            predictions,
        ],
        axis="columns",
    )

    outputs.columns = task.output_columns
    return outputs


def run_assistant(
    config_path: Annotated[
        str, typer.Argument(help="Path to the configuration directory, which includes a config.yaml file")
    ],
    task_path: Annotated[str, typer.Argument(help="Directory where task files are included")],
    config_overrides: Annotated[Optional[str], typer.Option(help="Overrides for the config in Hydra format")] = "",
) -> str:
    """Run AutoGluon-Assistant on a task defined in a path."""
    rel_config_path = _resolve_config_path(config_path)
    with initialize(version_base=None, config_path=rel_config_path):
        overrides_list = config_overrides.split(" ") if config_overrides else []
        config = compose(config_name="config", overrides=overrides_list)

    rprint("🤖 [bold red] Welcome to AutoGluon-Assistant [/bold red]")

    rprint("Will use task config:")
    rprint(OmegaConf.to_container(config))

    task_path = Path(task_path).resolve()
    rprint(f"Task path: {task_path}")

    task = TabularPredictionTask.from_path(task_path)

    rprint("[green]Task loaded![/green]")
    rprint(task)

    assistant = TabularPredictionAssistant(config)
    task = assistant.preprocess_task(task)

    rprint("[green]Task preprocessing complete![/green]")
    task_description = task.describe()

    for data_key in ["train_data", "test_data", "output_data"]:
        if data_key in task_description:
            rprint(f"{data_key}:")
            rprint(pd.DataFrame(task_description.pop(data_key, {})).T)

    rprint("[green]Task description:[/green]")
    rprint(task_description)

    assistant.fit_predictor(task)

    rprint("[green]Predictor fit complete![/green]")

    predictions = assistant.predict(task)

    output_filename = f"aga-output-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(output_filename, "w") as fp:
        make_prediction_outputs(task, predictions).to_csv(fp, index=False)

    rprint(f"[green]Prediction complete! Outputs written to {output_filename}[/green]")

    if config.save_artifacts.enabled:
        # Determine the filename with or without timestamp
        pickle_file_name = f"{task.name}_artifacts.pkl"
        if config.save_artifacts.append_timestamp:
            current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pickle_file_name = f"{task.name}_artifacts_{current_timestamp}.pkl"

        full_save_path = f"{config.save_artifacts.path.rstrip('/')}/{pickle_file_name}"

        task.save_artifacts(full_save_path, assistant.predictor, task.train_data, task.test_data, task.output_data)

        rprint(f"[green]Artifacts including transformed datasets and trained model saved at {full_save_path}[/green]")

    return output_filename


def main():
    app = typer.Typer(pretty_exceptions_enable=False)
    app.command()(run_assistant)
    app()


if __name__ == "__main__":
    main()
