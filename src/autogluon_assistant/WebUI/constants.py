from copy import deepcopy

BASE_DATA_DIR = "./user_data"


# Preset configurations
PRESET_DEFAULT_CONFIG = {
    "Best Quality": {"time_limit": "4 hrs", "feature_generation": True},
    "High Quality": {"time_limit": "1 hr", "feature_generation": False},
    "Medium Quality": {"time_limit": "10 mins", "feature_generation": False},
}
DEFAULT_PRESET = "Best Quality"

PRESET_MAPPING = {
    "Best Quality": "best_quality",
    "High Quality": "high_quality",
    "Medium Quality": "medium_quality",
}
PRESET_OPTIONS = ["Best Quality", "High Quality", "Medium Quality"]

# Time limit configurations (in seconds)
TIME_LIMIT_MAPPING = {
    "1 min": 60,
    "10 mins": 600,
    "30 mins": 1800,
    "1 hr": 3600,
    "2 hrs": 7200,
    "4 hrs": 14400,
}

DEFAULT_TIME_LIMIT = "4 hrs"

TIME_LIMIT_OPTIONS = ["1 min", "10 mins", "30 mins", "1 hr", "2 hrs", "4 hrs"]

# LLM configurations
LLM_MAPPING = {
    "Claude 3.5 with bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "GPT 4": "gpt-4o-mini-2024-07-18",
}

LLM_OPTIONS = ["Claude 3.5 with bedrock"]

# Provider configuration
PROVIDER_MAPPING = {"Claude 3.5 with bedrock": "bedrock", "GPT 4": "openai"}


API_KEY_LOCATION = {"Claude 3.5 with bedrock": "BEDROCK_API_KEY", "GPT 4": "OPENAI_API_KEY"}

INITIAL_STAGE = {
    "Task Understanding": [],
    "Feature Generation": [],
    "Model Training": [],
    "Prediction": [],
}
# Initial Session state
DEFAULT_SESSION_VALUES = {
    "config_overrides": [],
    "preset": DEFAULT_PRESET,
    "time_limit": DEFAULT_TIME_LIMIT,
    "llm": None,
    "pid": None,
    "logs": "",
    "process": None,
    "clicked": False,
    "task_running": False,
    "output_file": None,
    "output_filename": None,
    "task_description": "",
    "sample_description": "",
    "return_code": None,
    "task_canceled": False,
    "uploaded_files": {},
    "sample_files": {},
    "selected_dataset": None,
    "sample_dataset_dir": None,
    "description_uploader_key": 0,
    "sample_dataset_selector": None,
    "current_stage": None,
    "feature_generation": True,
    "stage_status": {},
    "show_remaining_time": False,
    "model_path": None,
    "increment_time": 0,
    "progress_bar": None,
    "increment": 2,
    "zip_path": None,
    "stage_container": deepcopy(INITIAL_STAGE),
}


# DataSet Options
DATASET_OPTIONS = ["Sample Dataset", "Upload Dataset"]

# Captions under DataSet Options
CAPTIONS = ["Run with sample dataset", "Upload Train, Test and Output (Optional) Dataset"]
