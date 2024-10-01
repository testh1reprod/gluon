# Autogluon Assistant

AutoGluon Assistant (AG-A) provides users a simple interface where they can input their data, describe their problem, and receive a highly accurate and competitive ML solution — without writing any code. By leveraging the state-of-the-art AutoML capabilities of AutoGluon and integrating them with a Large Language Model (LLM), AG-A automates the entire data science pipeline. AG-A takes AutoGluon’s automation from three lines of code to zero, enabling users to solve new supervised learning tabular problems using only natural language descriptions.

## Setup

```
# create a conda env
conda create -n aga python=3.10
conda activate aga

# clone repositories
git clone https://github.com/autogluon/autogluon-assistant.git
cd autogluon-assistant && pip install -e ".[dev]" && cd ..
```


### API Keys

#### LLMs 
You will need an OpenAI API key and have it set to the `OPENAI_API_KEY` environment variable.
- Create OpenAI Account: https://platform.openai.com/
- Manage OpenAI API Keys: https://platform.openai.com/account/api-keys

Note: If you have a free OpenAI account, then you will be blocked by capped rate limits.
	  The project requires paid OpenAI API keys access.

```
export OPENAI_API_KEY="sk-..."
```

You can also run AutoGluon-Assistant with Bedrock through the access gateway set up in the config, however `BEDROCK_API_KEY` will have to be present in the environment.
`autogluon-assistant-tools` provides more functionality and utilities for benchmarking, wrapped around autogluon-assistant. Please check out the [repo](https://github.com/autogluon/autogluon-assistant-tools/) for more details.


## Usage

Before launching AutoGluon Assistant (AG-A), ensure that the data files (dataset csv files, the dataset descriptions and task descriptions) are placed in the correct structure within the data directory. This setup is necessary for AG-A to run successfully. Below is a demo using a toy dataset from the Titanic Kaggle Competition.

```
.
├── config                            # Configuration files directory
│   └── [CONFIG_FILE].yaml            # Your configuration file
│
└── data                              # Data files directory
    ├── competition_files.txt         # Contains the names of all the files, for this case e.g. train.csv test.csv sample_submission.csv
    ├── data.txt                      # Contains the main data description
    ├── evaluation.txt                # Describes the evaluation metric to be used
    ├── sample_submission.csv         # Sample output template
    ├── test.csv                      # Test dataset
    └── train.csv                     # Training dataset

```

Now you can launch the AutoGluon Assistant run using the following command:
```
autogluon-assistant [NAME_OF_CONFIG_DIR] [NAME_OF_DATA_DIR]
# e.g. autogluon-assistant ./config ./toy_data
```

After the run is complete, model predictions on test dataset are saved into the `aga-output-<timestamp>.csv` file which is formatted according to `sample_submission.csv` file.

## Overriding parameters from the command-line
AutoGluon Assistant uses [Hydra](https://hydra.cc) to manage configuration. See [here](https://hydra.cc/docs/advanced/override_grammar/basic/) for the complete override syntax.
You can override specific settings in the YAML configuration defined in [`config.yaml`](https://github.com/autogluon/autogluon-assistant/blob/main/config/config.yaml) using
the `config_overrides` parameter with Hydra syntax from the command line.

Here’s an example command with some configuration overrides:
```
autogluon-assistant ./config ./data --output-filename my_output.csv --config-overrides "autogluon.predictor_fit_kwargs.time_limit=120 autogluon.predictor_fit_kwargs.verbosity=3 autogluon.predictor_fit_kwargs.presets=medium_quality llm.temperature=0.7 llm.max_tokens=256"
```
