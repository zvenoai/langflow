import json
from collections.abc import Sequence
from typing import Any

from langchain.agents import create_tool_calling_agent
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from lfx.base.agents.agent import LCToolsAgentComponent
from lfx.inputs.inputs import (
    DataInput,
    HandleInput,
    MessageTextInput,
)
from lfx.schema.data import Data


def _create_tool_message_with_reasoning(agent_action: ToolAgentAction, observation: str) -> ToolMessage:
    """Convert agent action and observation into a tool message.

    Preserves reasoning_details/thought_signature from the original message
    for OpenRouter/Gemini models compatibility.
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except (TypeError, ValueError):
            content = str(observation)
    else:
        content = observation

    additional_kwargs: dict[str, Any] = {"name": agent_action.tool}

    # Preserve reasoning_details and thought_signature from the original message
    if agent_action.message_log:
        for msg in agent_action.message_log:
            if isinstance(msg, AIMessage) and msg.additional_kwargs:
                # Preserve reasoning_details (OpenRouter format)
                if "reasoning_details" in msg.additional_kwargs:
                    additional_kwargs["reasoning_details"] = msg.additional_kwargs["reasoning_details"]

                # Preserve reasoning field
                if "reasoning" in msg.additional_kwargs:
                    additional_kwargs["reasoning"] = msg.additional_kwargs["reasoning"]

                # Check for thought_signature in additional_kwargs
                if "thought_signature" in msg.additional_kwargs:
                    additional_kwargs["thought_signature"] = msg.additional_kwargs["thought_signature"]

                # Check in tool_calls for thought_signature
                if "tool_calls" in msg.additional_kwargs:
                    for tc in msg.additional_kwargs.get("tool_calls", []):
                        if isinstance(tc, dict) and "thought_signature" in tc:
                            additional_kwargs["thought_signature"] = tc["thought_signature"]
                            break

    return ToolMessage(
        tool_call_id=agent_action.tool_call_id,
        content=content,
        additional_kwargs=additional_kwargs,
    )


def format_to_tool_messages_with_reasoning(
    intermediate_steps: Sequence[tuple[AgentAction, str]],
) -> list[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into ToolMessages.

    This version preserves reasoning_details for OpenRouter/Gemini models compatibility.
    When Gemini models use tool calling with "thinking" mode, they return reasoning_details
    that MUST be preserved and passed back in subsequent requests.

    See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks
    """
    messages: list[BaseMessage] = []
    for agent_action, observation in intermediate_steps:
        if isinstance(agent_action, ToolAgentAction):
            # Create tool message with reasoning preserved from the original AI message
            tool_message = _create_tool_message_with_reasoning(agent_action, observation)

            # Combine original AI messages from message_log with tool response
            # message_log can be None, so use fallback to empty list
            new_messages = [*(agent_action.message_log or []), tool_message]
            messages.extend([new for new in new_messages if new not in messages])
        else:
            messages.append(AIMessage(content=agent_action.log))
    return messages


class ToolCallingAgentComponent(LCToolsAgentComponent):
    display_name: str = "Tool Calling Agent"
    description: str = "An agent designed to utilize various tools seamlessly within workflows."
    icon = "LangChain"
    name = "ToolCallingAgent"

    inputs = [
        *LCToolsAgentComponent.get_base_inputs(),
        HandleInput(
            name="llm",
            display_name="Language Model",
            input_types=["LanguageModel"],
            required=True,
            info="Language model that the agent utilizes to perform tasks effectively.",
        ),
        MessageTextInput(
            name="system_prompt",
            display_name="System Prompt",
            info="System prompt to guide the agent's behavior.",
            value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
        ),
        DataInput(
            name="chat_history",
            display_name="Chat Memory",
            is_list=True,
            advanced=True,
            info="This input stores the chat history, allowing the agent to remember previous conversations.",
        ),
    ]

    def get_chat_history_data(self) -> list[Data] | None:
        return self.chat_history

    def create_agent_runnable(self):
        messages = [
            ("system", "{system_prompt}"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        self.validate_tool_names()
        try:
            # Use custom message formatter that preserves reasoning_details for OpenRouter/Gemini compatibility
            return create_tool_calling_agent(
                self.llm,
                self.tools or [],
                prompt,
                message_formatter=format_to_tool_messages_with_reasoning,
            )
        except NotImplementedError as e:
            message = f"{self.display_name} does not support tool calling. Please try using a compatible model."
            raise NotImplementedError(message) from e
