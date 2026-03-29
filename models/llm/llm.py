import logging
from collections.abc import Generator, Mapping
from typing import Any, Optional, Union, cast

import openai
import tiktoken
from openai import OpenAI

from dify_plugin import LargeLanguageModel
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)

from ..common_chatgpt_codex import CodexAuthenticationError, _CommonChatGPTCodex

logger = logging.getLogger(__name__)
AUTH_ERRORS = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    CodexAuthenticationError,
)


class ChatGPTCodexLargeLanguageModel(_CommonChatGPTCodex, LargeLanguageModel):
    """
    ChatGPT Codex large language model implementation.

    This provider intentionally supports only the ChatGPT-backed Codex Responses API.
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        model_parameters = model_parameters.copy()
        if model_parameters.pop("enable_stream", None) is False:
            stream = False

        prompt_messages = self._clear_illegal_prompt_messages(model, prompt_messages)

        if stream:
            return self._chat_generate_responses_api_stream(
                model=model,
                credentials=credentials,
                prompt_messages=prompt_messages,
                model_parameters=model_parameters,
                tools=tools,
                stop=stop,
                user=user,
            )

        return self._chat_generate_responses_api(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            user=user,
        )

    def validate_credentials(self, model: str, credentials: Mapping) -> None:
        """
        Validate credentials against the Codex Responses API only.
        """

        def _validate(client: OpenAI, _: dict) -> None:
            client.responses.create(
                model=model,
                input="ping",
                max_output_tokens=20,
            )

        try:
            self._with_codex_client(credentials, _validate)
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex)) from ex

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        return self._num_tokens_from_messages(model, prompt_messages, tools)

    def _with_codex_client(self, credentials: Mapping, func):
        current_credentials = dict(credentials)
        client = OpenAI(**self._to_credential_kwargs(current_credentials))

        try:
            return func(client, current_credentials)
        except AUTH_ERRORS:
            refreshed = self._refresh_access_token(current_credentials)
            current_credentials["chatgpt_access_token"] = refreshed["access_token"]
            current_credentials["chatgpt_refresh_token"] = refreshed["refresh_token"]
            if refreshed.get("account_id"):
                current_credentials["chatgpt_account_id"] = refreshed["account_id"]

            refreshed_client = OpenAI(**self._to_credential_kwargs(current_credentials))
            return func(refreshed_client, current_credentials)

    def _build_responses_api_params(
        self,
        model_parameters: dict,
        stop: Optional[list[str]] = None,
        user: Optional[str] = None,
    ) -> dict:
        params = model_parameters.copy()

        if "max_tokens" in params:
            params["max_output_tokens"] = params.pop("max_tokens")
        if "max_completion_tokens" in params:
            params["max_output_tokens"] = params.pop("max_completion_tokens")

        reasoning_effort = params.pop("reasoning_effort", None)
        if reasoning_effort and reasoning_effort != "none":
            params["reasoning"] = {"effort": reasoning_effort}

        response_format = params.pop("response_format", None)
        json_schema = params.pop("json_schema", None)
        if response_format == "json_schema" and json_schema:
            schema_obj = json_schema if isinstance(json_schema, dict) else {"schema": json_schema}
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_obj.get("name", "response"),
                    "schema": schema_obj.get("schema", json_schema),
                }
            }
            if "strict" in schema_obj:
                params["text"]["format"]["strict"] = schema_obj["strict"]

        if stop:
            params["stop"] = stop

        if user:
            params["user"] = user

        return params

    def _convert_prompt_messages_to_responses_input(
        self,
        prompt_messages: list[PromptMessage],
    ) -> list[dict]:
        input_items: list[dict] = []

        for message in prompt_messages:
            if isinstance(message, SystemPromptMessage):
                if isinstance(message.content, str):
                    content: Any = message.content
                else:
                    content = "\n".join(
                        item.data
                        for item in message.content
                        if item.type == PromptMessageContentType.TEXT
                    )
                input_items.append({"type": "message", "role": "system", "content": content})
                continue

            if isinstance(message, UserPromptMessage):
                if isinstance(message.content, str):
                    content = message.content
                else:
                    content = []
                    for item in message.content:
                        if item.type == PromptMessageContentType.TEXT:
                            text_item = cast(TextPromptMessageContent, item)
                            content.append({"type": "input_text", "text": text_item.data})
                        elif item.type == PromptMessageContentType.IMAGE:
                            image_item = cast(ImagePromptMessageContent, item)
                            content.append(
                                {
                                    "type": "input_image",
                                    "image_url": image_item.data,
                                    "detail": image_item.detail.value,
                                }
                            )
                input_items.append({"type": "message", "role": "user", "content": content})
                continue

            if isinstance(message, AssistantPromptMessage):
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        input_items.append(
                            {
                                "type": "function_call",
                                "call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            }
                        )
                else:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": message.content if isinstance(message.content, str) else "",
                        }
                    )
                continue

            if isinstance(message, ToolPromptMessage):
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content if isinstance(message.content, str) else "",
                    }
                )

        return input_items

    def _build_responses_api_tools(self, tools: Optional[list[PromptMessageTool]]) -> Optional[list[dict]]:
        if not tools:
            return None

        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]

    def _extract_output_tool_calls(self, output_items: list[Any]) -> list[AssistantPromptMessage.ToolCall]:
        tool_calls: list[AssistantPromptMessage.ToolCall] = []
        for item in output_items:
            if getattr(item, "type", None) != "function_call":
                continue
            tool_calls.append(
                AssistantPromptMessage.ToolCall(
                    id=getattr(item, "call_id", ""),
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=getattr(item, "name", ""),
                        arguments=getattr(item, "arguments", "") or "",
                    ),
                )
            )
        return tool_calls

    def _chat_generate_responses_api(
        self,
        model: str,
        credentials: Mapping,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]],
        stop: Optional[list[str]] = None,
        user: Optional[str] = None,
    ) -> LLMResult:
        response_params = self._build_responses_api_params(model_parameters, stop=stop, user=user)
        response_params["input"] = self._convert_prompt_messages_to_responses_input(prompt_messages)

        api_tools = self._build_responses_api_tools(tools)
        if api_tools:
            response_params["tools"] = api_tools
            response_params.setdefault("tool_choice", "auto")

        def _call(client: OpenAI, current_credentials: dict) -> LLMResult:
            response = client.responses.create(model=model, **response_params)

            assistant_prompt_message = AssistantPromptMessage(
                content=response.output_text or "",
                tool_calls=self._extract_output_tool_calls(getattr(response, "output", [])),
            )

            usage = None
            if response.usage:
                usage = self._calc_response_usage(
                    model=model,
                    credentials=current_credentials,
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                )

            return LLMResult(
                model=response.model,
                prompt_messages=prompt_messages,
                message=assistant_prompt_message,
                usage=usage,
                system_fingerprint=None,
            )

        return self._with_codex_client(credentials, _call)

    def _chat_generate_responses_api_stream(
        self,
        model: str,
        credentials: Mapping,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]],
        stop: Optional[list[str]] = None,
        user: Optional[str] = None,
    ) -> Generator:
        response_params = self._build_responses_api_params(model_parameters, stop=stop, user=user)
        response_params["input"] = self._convert_prompt_messages_to_responses_input(prompt_messages)

        api_tools = self._build_responses_api_tools(tools)
        if api_tools:
            response_params["tools"] = api_tools
            response_params.setdefault("tool_choice", "auto")

        stream_response = self._with_codex_client(
            credentials,
            lambda client, _: client.responses.create(model=model, stream=True, **response_params),
        )

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        final_model = model
        pending_tool_calls: dict[int, dict] = {}
        final_chunk = LLMResultChunk(
            model=model,
            prompt_messages=prompt_messages,
            delta=LLMResultChunkDelta(index=0, message=AssistantPromptMessage(content="")),
        )

        for event in stream_response:
            event_type = getattr(event, "type", "")
            logger.info("Codex Responses API stream event: %s", event_type)

            if event_type == "response.output_text.delta":
                delta_text = getattr(event, "delta", None) or getattr(event, "text", "") or ""
                if not delta_text:
                    continue
                full_text += delta_text
                yield LLMResultChunk(
                    model=final_model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=0,
                        message=AssistantPromptMessage(content=delta_text),
                    ),
                )
                continue

            if event_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "function_call":
                    pending_tool_calls[getattr(event, "output_index", 0)] = {
                        "call_id": getattr(item, "call_id", ""),
                        "name": getattr(item, "name", ""),
                        "arguments": "",
                    }
                continue

            if event_type == "response.function_call_arguments.delta":
                output_index = getattr(event, "output_index", 0)
                if output_index in pending_tool_calls:
                    pending_tool_calls[output_index]["arguments"] += getattr(event, "delta", "")
                continue

            if event_type == "response.function_call_arguments.done":
                output_index = getattr(event, "output_index", 0)
                if output_index in pending_tool_calls:
                    pending_tool_calls[output_index]["arguments"] = getattr(event, "arguments", "")
                    pending_tool_calls[output_index]["name"] = getattr(event, "name", "")
                continue

            if event_type == "response.completed":
                response = getattr(event, "response", None)
                if response is not None:
                    final_model = getattr(response, "model", final_model)
                    usage = getattr(response, "usage", None)
                    if usage:
                        prompt_tokens = usage.input_tokens
                        completion_tokens = usage.output_tokens
                    if not full_text and not pending_tool_calls:
                        full_text = getattr(response, "output_text", "") or ""
                        if full_text:
                            yield LLMResultChunk(
                                model=final_model,
                                prompt_messages=prompt_messages,
                                delta=LLMResultChunkDelta(
                                    index=0,
                                    message=AssistantPromptMessage(content=full_text),
                                ),
                            )

                if pending_tool_calls:
                    tool_calls = [
                        AssistantPromptMessage.ToolCall(
                            id=tool_call["call_id"],
                            type="function",
                            function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                name=tool_call["name"],
                                arguments=tool_call["arguments"],
                            ),
                        )
                        for tool_call in pending_tool_calls.values()
                    ]
                    yield LLMResultChunk(
                        model=final_model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(
                            index=0,
                            message=AssistantPromptMessage(content="", tool_calls=tool_calls),
                            finish_reason="tool_calls",
                        ),
                    )
                    final_chunk = LLMResultChunk(
                        model=final_model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(
                            index=0,
                            message=AssistantPromptMessage(content=""),
                            finish_reason="tool_calls",
                        ),
                    )
                else:
                    final_chunk = LLMResultChunk(
                        model=final_model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(
                            index=0,
                            message=AssistantPromptMessage(content=""),
                            finish_reason="stop",
                        ),
                    )

        if not prompt_tokens:
            prompt_tokens = self._num_tokens_from_messages(model, prompt_messages, tools)
        if not completion_tokens:
            completion_tokens = self._num_tokens_from_string(model, full_text)

        usage = self._calc_response_usage(
            model=model,
            credentials=dict(credentials),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        final_chunk.delta.usage = usage
        yield final_chunk

    def _clear_illegal_prompt_messages(
        self,
        model: str,
        prompt_messages: list[PromptMessage],
    ) -> list[PromptMessage]:
        return prompt_messages

    def _convert_prompt_message_to_dict(self, message: PromptMessage) -> dict:
        if isinstance(message, UserPromptMessage):
            if isinstance(message.content, str):
                message_dict = {"role": "user", "content": message.content}
            else:
                content_items = []
                for item in message.content:
                    if item.type == PromptMessageContentType.TEXT:
                        text_item = cast(TextPromptMessageContent, item)
                        content_items.append({"type": "text", "text": text_item.data})
                    elif item.type == PromptMessageContentType.IMAGE:
                        image_item = cast(ImagePromptMessageContent, item)
                        content_items.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_item.data,
                                    "detail": image_item.detail.value,
                                },
                            }
                        )
                message_dict = {"role": "user", "content": content_items}
        elif isinstance(message, AssistantPromptMessage):
            message_dict = {
                "role": "assistant",
                "content": message.content if isinstance(message.content, str) else "",
            }
            if message.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type or "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ]
        elif isinstance(message, SystemPromptMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, ToolPromptMessage):
            message_dict = {
                "role": "tool",
                "content": message.content,
                "tool_call_id": message.tool_call_id,
            }
        else:
            raise ValueError(f"Unsupported prompt message type: {type(message)}")

        if message.name and message_dict.get("role") != "tool":
            message_dict["name"] = message.name

        return message_dict

    def _num_tokens_from_string(
        self,
        model: str,
        text: str,
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = len(encoding.encode(text))
        if tools:
            num_tokens += self._num_tokens_for_tools(encoding, tools)
        return num_tokens

    def _num_tokens_from_messages(
        self,
        model: str,
        messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        if model.startswith("ft:"):
            model = model.split(":")[1]

        if model.startswith(("gpt-5", "o")):
            model = "gpt-4o"

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning("Model %s not found in tiktoken; falling back to cl100k_base.", model)
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0

        for message in [self._convert_prompt_message_to_dict(item) for item in messages]:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, list):
                    text = ""
                    for item in value:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text += item["text"]
                    value = text

                if key == "tool_calls":
                    for tool_call in value:  # type: ignore[assignment]
                        for tool_key, tool_value in tool_call.items():
                            num_tokens += len(encoding.encode(tool_key))
                            if tool_key == "function":
                                for function_key, function_value in tool_value.items():
                                    num_tokens += len(encoding.encode(function_key))
                                    num_tokens += len(encoding.encode(function_value))
                            else:
                                num_tokens += len(encoding.encode(str(tool_value)))
                else:
                    num_tokens += len(encoding.encode(str(value)))

                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3

        if tools:
            num_tokens += self._num_tokens_for_tools(encoding, tools)

        return num_tokens

    def _num_tokens_for_tools(self, encoding: tiktoken.Encoding, tools: list[PromptMessageTool]) -> int:
        num_tokens = 0
        for tool in tools:
            num_tokens += len(encoding.encode("type"))
            num_tokens += len(encoding.encode("function"))
            num_tokens += len(encoding.encode("name"))
            num_tokens += len(encoding.encode(tool.name))
            num_tokens += len(encoding.encode("description"))
            num_tokens += len(encoding.encode(tool.description or ""))
            num_tokens += len(encoding.encode("parameters"))
            num_tokens += len(encoding.encode(str(tool.parameters or {})))
        return num_tokens
