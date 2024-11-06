BASE_DATA_DIR = "./user_data"

# Preset configurations
PRESET_MAPPING = {
    "Best Quality": "best_quality",
    "High Quality": "high_quality",
    "Good Quality": "good_quality",
    "Medium Quality": "medium_quality",
}
PRESET_OPTIONS = ["Best Quality", "High Quality", "Good Quality", "Medium Quality"]

# Time limit configurations (in seconds)
TIME_LIMIT_MAPPING = {
    "1 min": 60,
    "15 mins": 900,
    "30 mins": 1800,
    "1 hr": 3600,
    "2 hrs": 7200,
    "4 hrs": 14400,
}
TIME_LIMIT_OPTIONS = ["1 min", "15 mins", "30 mins", "1 hr", "2 hrs", "4 hrs"]

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
# DataSet Options
DATASET_OPTIONS = ["Sample Dataset", "Upload Dataset"]

# Captions under DataSet Options
CAPTIONS = ["Run with sample dataset", "Upload Train, Test and Output (Optional) Dataset"]
