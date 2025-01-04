from typing import Optional, Any, Callable

from langchain_community.vectorstores.pgvector import BaseModel
from pydantic import ConfigDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableSerializable, RunnableConfig, Runnable
from langchain_core.runnables.utils import Input, Output
from langchain_core.tools import BaseTool

class RunnableWithTools(RunnableSerializable[Input, Output]):
    bound: Runnable[Input, Output]
    tools: dict[str, BaseTool]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        max_depth = 3
        depth = 0
        message = None
        while depth < max_depth:
            message = self.bound.invoke(input)
            if isinstance(message, AIMessage) and message.tool_calls and self.tools:
                text = ''
                if isinstance(message.content, list) and message.content[0]['text']:
                    text += message.content[0]['text']
                elif message.content:
                    text += message.content
                # input.append(AIMessage(content=text, **message.additional_kwargs))
                input.append(message)

                text = ''
                for tool_call in message.tool_calls:
                    selected_tool = self.tools[tool_call["name"].lower()]
                    if selected_tool:
                        tool_msg = selected_tool.invoke(tool_call)
                        if isinstance(tool_msg, BaseModel):
                            tool_msg = tool_msg.model_dump()
                        else:
                            tool_msg = str(tool_msg)
                        text += '\n' + tool_msg
                    input.append(ToolMessage(tool_call_id=tool_call['id'], content=text))
                depth += 1
            else:
                break

        return message

