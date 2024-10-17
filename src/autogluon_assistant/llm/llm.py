import os
import pprint
from typing import Any, Dict, List

from langchain_community.chat_models.bedrock import BedrockChat
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from omegaconf import DictConfig
from pydantic import Field


class AssistantChatOpenAI(ChatOpenAI):
    """
    AssistantChatOpenAI is a subclass of ChatOpenAI that traces the input and output of the model.
    """

    history_: List[Dict[str, Any]] = Field(default_factory=list) 
    input_: int = Field(default_factory=int)
    output_: int = Field(default_factory=int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_ = []
        self.input_ = 0     
        self.output_ = 0 

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "proxy": self.openai_proxy,
            "history": self.history_,
            "prompt_tokens": self.input_,
            "completion_tokens": self.output_,
        }

    def __call__(self, *args, **kwargs):
        input_: List[BaseMessage] = args[0]
        response = super().invoke(*args, **kwargs)
        self.input_ += int(response.response_metadata["token_usage"]['prompt_tokens'])
        self.output_ += response.response_metadata["token_usage"]['completion_tokens'] 
        self.history_.append(
            {
                "input": [{"type": msg.type, "content": msg.content} for msg in input_],
                "output": pprint.pformat(dict(response)),
                "prompt_tokens": self.input_,
                "completion_tokens": self.output_,
            }
        )
        return response


class AssistantChatBedrock(BedrockChat):
    """
    AssistantChatBedrock is a subclass of BedrockChat that traces the input and output of the model.
    """

    history_: List[Dict[str, Any]] = Field(default_factory=list)
    input_: int = Field(default_factory=int)
    output_: int = Field(default_factory=int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_ = []
        self.input_ = 0
        self.output_ = 0

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model_id,
            "history": self.history_,
            "prompt_tokens": self.input_,
            "completion_tokens": self.output_,
        }

    def __call__(self, *args, **kwargs):
        input_: List[BaseMessage] = args[0]
        response = super().invoke(*args, **kwargs)
        self.input_ += response.response_metadata["usage"]['prompt_tokens']
        self.output_ += response.response_metadata["usage"]['completion_tokens'] 
        self.history_.append(
            {
                "input": [{"type": msg.type, "content": msg.content} for msg in input_],
                "output": pprint.pformat(dict(response)),
            }
        )
        return response


class LLMFactory:
    valid_models = {
        "openai": ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4o-mini-2024-07-18"],
        "bedrock": ["anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-5-sonnet-20240620-v1:0"],
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
    def get_chat_model(config: DictConfig) -> AssistantChatOpenAI:
        assert config.provider in LLMFactory.valid_models
        assert config.model in LLMFactory.valid_models[config.provider]

        if config.provider == "openai":
            return LLMFactory._get_openai_chat_model(config)
        elif config.provider == "bedrock":
            return LLMFactory._get_bedrock_chat_model(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
