import openai
import boto3

def get_openai_models():
    try:
        models = openai.Model.list()

        return [model.id for model in models.data]
    except Exception as e:
        print(f"Error fetching OpenAI models: {e}")
        return []

def get_bedrock_models():
    try:
        bedrock = boto3.client('bedrock')
        response = bedrock.list_foundation_models()

        return [model['modelId'] for model in response['modelSummaries']]
    except Exception as e:
        print(f"Error fetching Bedrock models: {e}")
        return []

def get_models_list(provider):
    if provider == "openai":
        return get_openai_models()
    elif provider == "bedrock":
        return get_bedrock_models()
    else:
        raise ValueError(f"Invalid LLM provider: {provider}")
