import random
from typing import TypedDict, Literal, NotRequired

from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.constants import START, END

from langgraph.graph import StateGraph


class PublicState(TypedDict):
    foo: int

class PrivateState(TypedDict):
    baz: int

if __name__ == '__main__':
    def node_1(state: PublicState) -> PrivateState:
        print("--- Node 1 ---")
        return {'baz': state['foo'] + 1}

    def node_2(state: PrivateState) -> PublicState:
        print("--- Node 2 ---")
        return {'foo': state['baz'] + 1}

    builder = StateGraph(PublicState)
    builder.add_node('node_1', node_1)
    builder.add_node('node_2', node_2)
    builder.add_edge(START, 'node_1')
    builder.add_edge('node_1', 'node_2')
    builder.add_edge('node_2', END)

    graph = builder.compile()

    state = graph.invoke({'foo': 1})
    print(state['foo'])
