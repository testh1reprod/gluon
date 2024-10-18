import os
import pprint
from typing import Any, Dict, List, Union

import boto3
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain.schema import BaseMessage, AIMessage
from omegaconf import DictConfig
from openai import OpenAI
from pydantic import Field, BaseModel


class AssistantChatOpenAI(ChatOpenAI, BaseModel):
    """
    AssistantChatOpenAI is a subclass of ChatOpenAI that traces the input and output of the model.
    """

    history_: List[Dict[str, Any]] = Field(default_factory=list)
    input_: int = Field(default=0)
    output_: int = Field(default=0)

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "proxy": self.openai_proxy,
            "history": self.history_,
            "prompt_tokens": self.input_,
            "completion_tokens": self.output_,
        }

    def invoke(self, *args, **kwargs):
        input_: List[BaseMessage] = args[0]
        response = super().invoke(*args, **kwargs)

        # Update token usage
        if isinstance(response, AIMessage) and response.usage_metadata:
            self.input_ += response.usage_metadata.get("input_tokens", 0)
            self.output_ += response.usage_metadata.get("output_tokens", 0)

        self.history_.append(
            {
                "input": [{"type": msg.type, "content": msg.content} for msg in input_],
                "output": pprint.pformat(dict(response)),
                "prompt_tokens": self.input_,
                "completion_tokens": self.output_,
            }
        )
        return response


class AssistantChatBedrock(ChatBedrock, BaseModel):
    """
    AssistantChatBedrock is a subclass of ChatBedrock that traces the input and output of the model.
    """

    history_: List[Dict[str, Any]] = Field(default_factory=list)
    input_: int = Field(default=0)
    output_: int = Field(default=0)

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model_id,
            "history": self.history_,
            "prompt_tokens": self.input_,
            "completion_tokens": self.output_,
        }

    def invoke(self, *args, **kwargs):
        input_: List[BaseMessage] = args[0]
        response = super().invoke(*args, **kwargs)

        # Update token usage
        if isinstance(response, AIMessage) and response.usage_metadata:
            self.input_ += response.usage_metadata.get("input_tokens", 0)
            self.output_ += response.usage_metadata.get("output_tokens", 0)

        self.history_.append(
            {
                "input": [{"type": msg.type, "content": msg.content} for msg in input_],
                "output": pprint.pformat(dict(response)),
                "prompt_tokens": self.input_,
                "completion_tokens": self.output_,
            }
        )
        return response


class LLMFactory:
    @staticmethod
    def get_openai_models() -> List[str]:
        try:
            client = OpenAI()
            models = client.models.list()
            return [model.id for model in models if model.id.startswith(("gpt-3.5", "gpt-4"))]
        except Exception as e:
            print(f"Error fetching OpenAI models: {e}")
            return []

    @staticmethod
    def get_bedrock_models() -> List[str]:
        try:
            bedrock = boto3.client('bedrock')
            response = bedrock.list_foundation_models()
            return [model['modelId'] for model in response['modelSummaries'] if model['modelId'].startswith("anthropic.claude")]
        except Exception as e:
            print(f"Error fetching Bedrock models: {e}")
            return []

    @classmethod
    def get_valid_models(cls):
        return {
            "openai": cls.get_openai_models(),
            "bedrock": cls.get_bedrock_models(),
        }

    @staticmethod
    def _get_openai_chat_model(config: DictConfig) -> AssistantChatOpenAI:
        if config.api_key_location in os.environ:
            api_key = os.environ[config.api_key_location]
        else:
            raise Exception("OpenAI API env variable not set")

        return AssistantChatOpenAI(
            model_name=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            verbose=config.verbose,
            openai_api_key=api_key,
            openai_api_base=config.proxy_url,
        )

    @staticmethod
    def _get_bedrock_chat_model(config: DictConfig) -> AssistantChatBedrock:
        return AssistantChatBedrock(
            model_id=config.model,
            model_kwargs={
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            },
            region_name=os.getenv("AWS_REGION"),
            verbose=config.verbose,
        )

    @classmethod
    def get_chat_model(cls, config: DictConfig) -> AssistantChatOpenAI | AssistantChatBedrock:
        valid_models = cls.get_valid_models()

        assert config.provider in valid_models
        assert config.model in valid_models[config.provider]

        if config.provider == "openai":
            return LLMFactory._get_openai_chat_model(config)
        elif config.provider == "bedrock":
            return LLMFactory._get_bedrock_chat_model(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
