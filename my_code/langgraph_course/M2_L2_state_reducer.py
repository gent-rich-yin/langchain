import random
from operator import add
from typing import TypedDict, Literal, NotRequired, List, Annotated

from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.constants import START, END

from langgraph.graph import StateGraph


class State(TypedDict):
    numbers: Annotated[List[int], add]

if __name__ == '__main__':
    def node_1(state: State) -> State:
        print("--- Node 1 ---")
        return {'numbers': [state['numbers'][0] + 1]}

    def node_2(state: State) -> State:
        print("--- Node 2 ---")
        return {'numbers': [state['numbers'][-1] + 1]}

    def node_3(state: State) -> State:
        print("--- Node 3 ---")
        return {'numbers': [state['numbers'][-1] + 1]}

    builder = StateGraph(State)
    builder.add_node('node_1', node_1)
    builder.add_node('node_2', node_2)
    builder.add_node('node_3', node_3)
    builder.add_edge(START, 'node_1')
    builder.add_edge('node_1', 'node_2')
    builder.add_edge('node_1', 'node_3')
    builder.add_edge('node_2', END)
    builder.add_edge('node_3', END)

    graph = builder.compile()

    state = graph.invoke({'numbers': [1]})
    print(state['numbers'])
