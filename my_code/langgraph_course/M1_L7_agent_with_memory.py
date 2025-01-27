import random
from typing import TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from langgraph.prebuilt.tool_node import tools_condition

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import START, END

from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


def multiply(a: float, b: float) -> float:
    """
    multiplies 2 numbers and return result
    """
    return a * b

def add(a: float, b: float) -> float:
    """
    adds 2 numbers and return result
    """
    return a + b

def divide(a: float, b: float) -> float:
    """
    divide a by b and return result
    """
    return a / b

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
    tools = [multiply, add, divide]
    model_with_tools = model.bind_tools(tools)

    system_message = SystemMessage(content='You are an assistant that can perform arithmetic calculations.')

    def assistant(state: MessagesState):
        return {'messages': model_with_tools.invoke([system_message] + state['messages'])}

    builder = StateGraph(MessagesState)
    builder.add_node('assistant', assistant)
    builder.add_node('tools', ToolNode(tools))

    builder.add_edge(START, 'assistant')
    builder.add_conditional_edges('assistant', tools_condition)
    builder.add_edge('tools', 'assistant')

    memory = MemorySaver()
    config = {'configurable': {'thread_id': str(uuid4())}}
    graph = builder.compile(checkpointer=memory)

    graph.invoke({'messages': [HumanMessage(content="multiple 2 and 3")]}, config)
    graph.invoke({'messages': [HumanMessage(content="add 4 to that result")]}, config)
    state = graph.invoke({'messages': [HumanMessage(content="divide that by 2")]}, config)
    for message in state['messages']:
        print(message)
