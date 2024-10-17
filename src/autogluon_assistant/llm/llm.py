import os
import pprint
from typing import Any, Dict, List, Union

from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain.schema import BaseMessage, AIMessage
from omegaconf import DictConfig
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
    valid_models = {
        "openai": ["gpt-3.5-turbo", "gpt-4-1106-preview"],
        "bedrock": [
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
        ],
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

    @staticmethod
    def get_chat_model(config: DictConfig) -> AssistantChatOpenAI | AssistantChatBedrock:
        assert config.provider in LLMFactory.valid_models
        assert config.model in LLMFactory.valid_models[config.provider]

        if config.provider == "openai":
            return LLMFactory._get_openai_chat_model(config)
        elif config.provider == "bedrock":
            return LLMFactory._get_bedrock_chat_model(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
