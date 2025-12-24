"""Custom ChatOpenAI wrapper that preserves reasoning_details for OpenRouter/Gemini compatibility.

This module provides a ChatOpenAI subclass that properly handles the `reasoning_details` field
required by Gemini models via OpenRouter for tool calling. Without this, Gemini models will fail
with "Function call is missing a thought_signature" errors.

See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI


def _convert_message_to_dict_with_reasoning(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary, preserving reasoning_details.

    This extends the standard LangChain conversion to include reasoning_details
    which is required by OpenRouter for Gemini models with tool calling.
    """
    from langchain_core.messages import (
        AIMessage,
        ChatMessage,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    # Format content
    content = message.content
    if isinstance(content, str):
        formatted_content = content
    elif isinstance(content, list):
        formatted_content = []
        for item in content:
            if isinstance(item, str):
                formatted_content.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                formatted_content.append(item)
        formatted_content = formatted_content if formatted_content else ""
    else:
        formatted_content = content

    message_dict: dict[str, Any] = {"content": formatted_content}

    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # Populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"

        # Handle tool calls
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = []
            for tc in message.tool_calls:
                tool_call_dict = {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("args", {}),
                    },
                }
                # Convert args to JSON string if it's a dict
                if isinstance(tool_call_dict["function"]["arguments"], dict):
                    tool_call_dict["function"]["arguments"] = json.dumps(tool_call_dict["function"]["arguments"])
                message_dict["tool_calls"].append(tool_call_dict)

            for tc in message.invalid_tool_calls:
                message_dict["tool_calls"].append(
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": tc.get("args", ""),
                        },
                    }
                )
        elif "tool_calls" in message.additional_kwargs:
            # Preserve raw tool_calls from additional_kwargs
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]

        # If tool calls present, content null value should be None not empty string
        if "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None

        # IMPORTANT: Preserve reasoning_details for OpenRouter/Gemini compatibility
        if "reasoning_details" in message.additional_kwargs:
            message_dict["reasoning_details"] = message.additional_kwargs["reasoning_details"]

        # Also preserve raw reasoning field
        if "reasoning" in message.additional_kwargs:
            message_dict["reasoning"] = message.additional_kwargs["reasoning"]

    elif isinstance(message, SystemMessage):
        message_dict["role"] = message.additional_kwargs.get("__openai_role__", "system")
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id
    else:
        msg = f"Got unknown message type: {type(message)}"
        raise TypeError(msg)

    return message_dict


def _convert_dict_to_message_with_reasoning(response_message: dict) -> AIMessage:
    """Convert API response dict to AIMessage, preserving reasoning_details.

    This ensures that reasoning_details from OpenRouter is stored in additional_kwargs
    so it can be passed back in subsequent requests.
    """
    from langchain_core.messages import ToolCall

    content = response_message.get("content", "") or ""
    additional_kwargs: dict[str, Any] = {}

    # Parse tool calls
    tool_calls = []
    invalid_tool_calls = []
    if raw_tool_calls := response_message.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        for raw_tool_call in raw_tool_calls:
            try:
                function_data = raw_tool_call.get("function", {})
                args = function_data.get("arguments", "{}")
                if isinstance(args, str):
                    args = json.loads(args) if args else {}

                tool_calls.append(
                    ToolCall(
                        name=function_data.get("name", ""),
                        args=args,
                        id=raw_tool_call.get("id", ""),
                    )
                )
            except (json.JSONDecodeError, TypeError, KeyError):
                invalid_tool_calls.append(
                    {
                        "name": raw_tool_call.get("function", {}).get("name"),
                        "args": raw_tool_call.get("function", {}).get("arguments"),
                        "id": raw_tool_call.get("id"),
                        "error": "Failed to parse tool call",
                    }
                )

    # IMPORTANT: Preserve reasoning_details for OpenRouter/Gemini
    if reasoning_details := response_message.get("reasoning_details"):
        additional_kwargs["reasoning_details"] = reasoning_details

    # Also preserve reasoning field
    if reasoning := response_message.get("reasoning"):
        additional_kwargs["reasoning"] = reasoning

    return AIMessage(
        content=content,
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
        id=response_message.get("id"),
    )


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that preserves reasoning_details for OpenRouter compatibility.

    This class extends ChatOpenAI to properly handle the `reasoning_details` field
    that is required by Gemini models via OpenRouter when using tool calling.

    When Gemini models use "thinking" mode with tool calls, they return a
    `reasoning_details` field that contains thought signatures. These signatures
    MUST be preserved and passed back in subsequent requests, otherwise the API
    will return error 400: "Function call is missing a thought_signature".

    Usage:
        model = ReasoningChatOpenAI(
            model="google/gemini-2.0-flash-thinking-exp",
            openai_api_key="...",
            openai_api_base="https://openrouter.ai/api/v1",
        )
    """

    def _create_message_dicts(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
    ) -> tuple[list[dict], dict]:
        """Override to use custom message conversion that preserves reasoning_details."""
        params = self._default_params
        if stop is not None:
            params["stop"] = stop

        # Use our custom conversion that preserves reasoning_details
        message_dicts = [_convert_message_to_dict_with_reasoning(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ) -> Any:
        """Override to preserve reasoning_details in the parsed message."""
        from langchain_core.outputs import ChatGeneration, ChatResult

        response_dict = response if isinstance(response, dict) else response.model_dump()

        if response_dict.get("error"):
            raise ValueError(response_dict["error"])

        generations = []
        for choice in response_dict.get("choices", []):
            message_dict = choice.get("message", {})

            # Use our custom conversion that preserves reasoning_details
            message = _convert_dict_to_message_with_reasoning(message_dict)

            gen_info = {
                "finish_reason": choice.get("finish_reason"),
                **(generation_info or {}),
            }

            generations.append(ChatGeneration(message=message, generation_info=gen_info))

        token_usage = response_dict.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": response_dict.get("model", self.model_name),
            "system_fingerprint": response_dict.get("system_fingerprint"),
        }

        return ChatResult(generations=generations, llm_output=llm_output)
