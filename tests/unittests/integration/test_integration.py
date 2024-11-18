import os

import pandas as pd
import pytest
from omegaconf import OmegaConf

from autogluon.assistant import run_assistant


@pytest.fixture
def titanic_data_path(tmp_path):
    # Create data directory
    data_dir = tmp_path / "titanic_data"
    data_dir.mkdir()

    # Download and save train/test data
    train_url = "https://autogluon.s3.amazonaws.com/datasets/titanic/train.csv"
    test_url = "https://autogluon.s3.amazonaws.com/datasets/titanic/test.csv"

    pd.read_csv(train_url).to_csv(data_dir / "train.csv", index=False)
    pd.read_csv(test_url).to_csv(data_dir / "test.csv", index=False)

    # Create description file
    description = """
    Binary classification task to predict passenger survival on the Titanic.

    Target Variable:
    - Survived: Survival (0 = No; 1 = Yes)

    Features include:
    - Pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
    - Sex: Gender
    - Age: Age in years
    - SibSp: Number of siblings/spouses aboard
    - Parch: Number of parents/children aboard
    - Fare: Passenger fare
    - Embarked: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

    Evaluation metric: Binary classification accuracy
    """

    with open(data_dir / "descriptions.txt", "w") as f:
        f.write(description)

    return str(data_dir)


@pytest.fixture
def light_config():
    return OmegaConf.create(
        {
            "time_limit": 300,  # 5 minutes timeout
            "llm": {"provider": "bedrock", "model": "anthropic.claude-3-5-haiku-20241022-v1:0"},
            "autogluon": {
                "predictor_fit_kwargs": {
                    "presets": "medium_quality",  # lighter preset
                }
            },
        }
    )


def test_titanic_prediction(titanic_data_path, light_config):
    # Convert config to string overrides
    config_overrides = [
        f"llm.provider={light_config.llm.provider}",
        f"llm.model={light_config.llm.model}",
        f"autogluon.predictor_fit_kwargs.presets={light_config.autogluon.predictor_fit_kwargs.presets}",
        f"time_limit={light_config.time_limit}",
        "feature_transformers.enabled_models=null",
    ]

    # Run assistant with config overrides
    output_file = run_assistant(
        task_path=titanic_data_path, presets="medium_quality", config_overrides=config_overrides
    )

    # Load original test data and predictions
    test_data = pd.read_csv(os.path.join(titanic_data_path, "test.csv"))
    predictions = pd.read_csv(output_file)

    # Basic validation checks
    assert os.path.exists(output_file)
    assert "Survived" in predictions.columns
    assert len(predictions) == len(test_data)
    assert predictions["Survived"].isin([0, 1]).all()
