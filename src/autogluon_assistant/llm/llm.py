import os
import pprint
from typing import Any, Dict, List

from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from omegaconf import DictConfig
from pydantic import Field


class AssistantChatOpenAI(ChatOpenAI):
    """
    AssistantChatOpenAI is a subclass of ChatOpenAI that traces the input and output of the model.
    """

    history_: List[Dict[str, Any]] = Field(default_factory=list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_ = []

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "proxy": self.openai_proxy,
            "history": self.history_,
        }

    def __call__(self, *args, **kwargs):
        input_: List[BaseMessage] = args[0]
        response = super().__call__(*args, **kwargs)

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
        "bedrock": ["anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-haiku-20240307-v1:0"],
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
    def get_chat_model(config: DictConfig) -> AssistantChatOpenAI:
        assert config.provider in LLMFactory.valid_models
        assert config.model in LLMFactory.valid_models[config.provider]

        return LLMFactory._get_openai_chat_model(config)
