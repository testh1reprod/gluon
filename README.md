# Autogluon Assistant

[![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.org/project/autogluon-assistant/)
[![GitHub license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Continuous Integration](https://github.com/autogluon/autogluon-assistant/actions/workflows/lint.yml/badge.svg)](https://github.com/autogluon/autogluon-assistant/actions/workflows/lint.yml)

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

#### Configuring LLMs
AG-A supports using both AWS Bedrock and OpenAI as LLM model providers. You will need to set up API keys for the respective provider you choose. By default, AG-A uses AWS Bedrock for its language models.

#### AWS Bedrock Setup
AG-A integrates with AWS Bedrock by default. To use AWS Bedrock, you will need to configure your AWS credentials and region settings, along with the Bedrock-specific API key:

```bash
export BEDROCK_API_KEY="4509..."
export AWS_DEFAULT_REGION="<your-region>"
export AWS_ACCESS_KEY_ID="<your-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```

Ensure you have an active AWS account and appropriate permissions set up for Bedrock. You can manage your AWS credentials through the AWS Management Console.


#### OpenAI Setup
To use OpenAI, you'll need to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

You can sign up for an OpenAI account [here](https://platform.openai.com/) and manage your API keys [here](https://platform.openai.com/account/api-keys).

Important: Free-tier OpenAI accounts may be subject to rate limits, which could affect AG-A's performance. We recommend using a paid OpenAI API key for seamless functionality.


## Usage
Before launching AutoGluon Assistant (AG-A), prepare your data files in the following structure. Here's an example using a dataset from the Titanic Kaggle Competition:
```
.
├── config # Configuration files directory
│ └── [CONFIG_FILE].yaml # Your configuration file
│
└── data # Data files directory
    ├── train.[ext] # Training dataset (required)
    ├── test.[ext]  # Test dataset (required)
    └── description.txt # Dataset and task description (recommended)
```
Note:
- The training and test files can be in any tabular data format (e.g., csv, parquet, xlsx)
- While there are no strict naming requirements, we recommend using clear, descriptive filenames
- The description file is optional but recommended for better model selection and optimization. It can include:
  - Dataset description
  - Problem context
  - Evaluation metrics
  - Any other relevant information

Now you can launch the AutoGluon Assistant run using the following command:
```
aga [NAME_OF_DATA_DIR]
# e.g. aga ./toy_data
```

After the run is complete, model predictions on test dataset are saved into the `aga-output-<timestamp>.csv` file which is formatted according to `sample_submission.csv` file.

## Overriding parameters from the command-line
AutoGluon Assistant uses [Hydra](https://hydra.cc) to manage configuration. See [here](https://hydra.cc/docs/advanced/override_grammar/basic/) for the complete override syntax.
You can override specific settings in the YAML configuration defined in [`config.yaml`](https://github.com/autogluon/autogluon-assistant/blob/main/config/config.yaml) using
the `config_overrides` parameter with Hydra syntax from the command line.

Here’s an example command with some configuration overrides:
```
autogluon-assistant ./data ./config --output-filename my_output.csv --config-overrides "autogluon.predictor_fit_kwargs.time_limit=120 autogluon.predictor_fit_kwargs.verbosity=3 autogluon.predictor_fit_kwargs.presets=medium_quality llm.temperature=0.7 llm.max_tokens=256"

# OR

aga ./data ./config --output-filename my_output.csv --config-overrides "autogluon.predictor_fit_kwargs.time_limit=120 autogluon.predictor_fit_kwargs.verbosity=3 autogluon.predictor_fit_kwargs.presets=medium_quality llm.temperature=0.7 llm.max_tokens=256"
```

`autogluon-assistant-tools` provides more functionality and utilities for benchmarking, wrapped around autogluon-assistant. Please check out the [repo](https://github.com/autogluon/autogluon-assistant-tools/) for more details.
