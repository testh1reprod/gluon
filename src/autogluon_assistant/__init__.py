import datetime
import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from omegaconf import OmegaConf
from rich import print as rprint
from typing_extensions import Annotated

from .assistant import TabularPredictionAssistant
from .constants import DEFAULT_QUALITY, NO_ID_COLUMN_IDENTIFIED, PRESETS
from .task import TabularPredictionTask
from .utils import load_config

logging.basicConfig(level=logging.INFO)

__all__ = ["TabularPredictionAssistant", "TabularPredictionTask"]


def get_task(path: Path) -> TabularPredictionTask:
    """Get a task from a path."""

    return TabularPredictionTask.from_path(path)


def make_prediction_outputs(task: TabularPredictionTask, predictions: pd.DataFrame) -> pd.DataFrame:
    # Convert predictions to DataFrame if it's a Series
    if isinstance(predictions, pd.Series):
        outputs = predictions.to_frame()
    else:
        outputs = predictions.copy()

    # Ensure we only keep required output columns from predictions
    common_cols = [col for col in task.output_columns if col in outputs.columns]
    outputs = outputs[common_cols]

    # Handle specific test ID column if provided and detected
    if task.test_id_column is not None and task.test_id_column != NO_ID_COLUMN_IDENTIFIED:
        test_ids = task.test_data[task.test_id_column]
        output_ids = task.sample_submission_data[task.output_id_column]

        if not test_ids.equals(output_ids):
            print(f"Warning: Test IDs and output IDs do not match!")

        # Ensure test ID column is included
        if task.test_id_column not in outputs.columns:
            outputs = pd.concat([task.test_data[task.test_id_column], outputs], axis="columns")

    # Handle undetected ID columns
    missing_columns = [col for col in task.output_columns if col not in outputs.columns]
    if missing_columns:
        print(
            f"Warning: The following columns are not in predictions and will be treated as ID columns: {missing_columns}"
        )

        for col in missing_columns:
            if col in task.test_data.columns:
                # Copy from test data if available
                outputs[col] = task.test_data[col]
                print(f"Warning: Copied from test data for column '{col}'")
            else:
                # Generate unique integer values
                outputs[col] = range(len(outputs))
                print(f"Warning: Generated unique integer values for column '{col}' as it was not found in test data")

    # Ensure columns are in the correct order
    outputs = outputs[task.output_columns]

    return outputs


def run_assistant(
    task_path: Annotated[str, typer.Argument(help="Directory where task files are included")],
    presets: Annotated[
        Optional[str],
        typer.Option("--presets", "-p", help="Presets"),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option("--config-path", "-c", help="Path to the configuration file (config.yaml)"),
    ] = None,
    config_overrides: Annotated[
        Optional[List[str]],
        typer.Option(
            "--config_overrides",
            "-o",
            help="Override config values. Format: key=value or key.nested=value. Can be used multiple times.",
        ),
    ] = None,
    output_filename: Annotated[Optional[str], typer.Option(help="Output File")] = "",
) -> str:
    logging.info("Starting AutoGluon-Assistant")

    if presets is None or presets not in PRESETS:
        logging.info(f"Presets is not provided or invalid: {presets}")
        presets = DEFAULT_QUALITY
        logging.info(f"Using default presets: {presets}")
    logging.info(f"Presets: {presets}")

    # Load config with all overrides
    try:
        config = load_config(presets, config_path, config_overrides)
        logging.info("Successfully loaded config")
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise

    rprint("🤖 [bold red] Welcome to AutoGluon-Assistant [/bold red]")

    rprint("Will use task config:")
    rprint(OmegaConf.to_container(config))

    task_path = Path(task_path).resolve()
    assert task_path.is_dir(), "Task path does not exist, please provide a valid directory."
    rprint(f"Task path: {task_path}")

    task = TabularPredictionTask.from_path(task_path)

    rprint("[green]Task loaded![/green]")
    rprint(task)

    assistant = TabularPredictionAssistant(config)
    task = assistant.preprocess_task(task)

    rprint("Predictor fit starts!")

    assistant.fit_predictor(task)

    rprint("[green]Predictor fit complete![/green]")

    rprint("Prediction starts!")

    predictions = assistant.predict(task)

    if not output_filename:
        output_filename = f"aga-output-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(output_filename, "w") as fp:
        make_prediction_outputs(task, predictions).to_csv(fp, index=False)

    rprint(f"[green]Prediction complete! Outputs written to {output_filename}[/green]")

    if config.save_artifacts.enabled:
        # Determine the artifacts_dir with or without timestamp
        artifacts_dir_name = f"{task.metadata['name']}_artifacts"
        if config.save_artifacts.append_timestamp:
            current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            artifacts_dir_name = f"{task.metadata['name']}_artifacts_{current_timestamp}"

        full_save_path = f"{config.save_artifacts.path.rstrip('/')}/{artifacts_dir_name}"

        task.save_artifacts(
            full_save_path, assistant.predictor, task.train_data, task.test_data, task.sample_submission_data
        )

        rprint(f"[green]Artifacts including transformed datasets and trained model saved at {full_save_path}[/green]")

    return output_filename


def main():
    app = typer.Typer(pretty_exceptions_enable=False)
    app.command()(run_assistant)
    app()


if __name__ == "__main__":
    main()
