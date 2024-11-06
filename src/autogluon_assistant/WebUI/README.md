# Autogluon Assistant Web UI

The Autogluon Assistant Web UI is a user-friendly application that allows users to leverage the capabilities of the Autogluon-Assistant library through an intuitive web interface.

## Features

1. **Dataset Upload and Task Definition**: Users can upload one or more CSV/XLSX files as their dataset and provide a free-form text description of the task they want to solve.
2. **Data Preview**: The web UI allows users to preview the uploaded datasets, enabling them to inspect the data before initiating an Autogluon-Assistant run.
3. **Autogluon-Assistant Configuration**: Users can configure the Autogluon-Assistant run through an intuitive user interface, with the ability to customize the parameters defined in the YAML configuration file.
4. **Run Execution and Monitoring**: Users can initiate the Autogluon-Assistant run and track the progress of the pipeline execution, with the help of status updates and visual indicators like progress bars.
5. **Results Reporting**: After the run is complete, users can view the evaluation metrics, download the submission file (predictions), and optionally download the trained Autogluon model.
6. **Multi-Session Support**: The web UI supports multiple user sessions, ensuring that each user's actions and data are isolated and secure, enabling concurrent users.

## API Keys
### LLMs 
You will need an OpenAI API key and have it set to the `OPENAI_API_KEY` environment variable.
- Create OpenAI Account: https://platform.openai.com/
- Manage OpenAI API Keys: https://platform.openai.com/account/api-keys

Note: If you have a free OpenAI account, then you will be blocked by capped rate limits.
	  The project requires paid OpenAI API keys access.

```
export OPENAI_API_KEY="sk-..."
```

You can also run AutoGluon-Assistant with Bedrock through the access gateway set up in the config, however `BEDROCK_API_KEY` will have to be present in the environment.
To run AutoGluon-Assistant with a local LLM, see [this section](#local-llm-support), but note that the `OPENAI_API_KEY` environment variable will still need to be set despite the value being unused.
## Installation

To install and run the Autogluon Assistant Web UI, follow these steps:

1. create virtual env
````
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U setuptools wheel
````
2. Clone the repository:
````
git clone https://github.com/autogluon/autogluon-assistant-tools.git
````
3. Navigate to the project directory:
````
cd autogluon-assistant-WebUI/src
````
4. Install the required dependencies:
````
pip install .
````
Alternatively, to install in editable mode during development:
````
pip install -e .
````
5. Run the Streamlit app:
````
streamlit run app.py
````
6. The Autogluon Assistant Web UI should now be accessible in your web browser at `http://localhost:8501`
