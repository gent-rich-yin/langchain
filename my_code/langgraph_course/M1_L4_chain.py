import random
from typing import TypedDict

from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.constants import START, END

from langgraph.graph import StateGraph, MessagesState

def multiply(a: float, b: float) -> float:
    """
    multiplies 2 numbers and return result
    """
    return a * b

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
    tools = [multiply]
    model_with_tools = model.bind_tools(tools)

    def tool_calling_llm(state: MessagesState):
        return {'messages': [model_with_tools.invoke(state['messages'])]}

    builder = StateGraph(MessagesState)
    builder.add_node('model', tool_calling_llm)

    builder.add_edge(START, 'model')
    builder.add_edge('model', END)

    graph = builder.compile()

    state = {'messages': [HumanMessage(content="multiple 2 and 3")]}
    state = graph.invoke(state)
    print(state)
