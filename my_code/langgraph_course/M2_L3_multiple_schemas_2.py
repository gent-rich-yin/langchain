import random
from typing import TypedDict, Literal, NotRequired

from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.constants import START, END

from langgraph.graph import StateGraph


class OverallState(TypedDict):
    _question: str
    _answer: str
    _notes: str

class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

if __name__ == '__main__':
    def node_1(state: InputState) -> OverallState:
        print("--- Node 1 ---")
        return {
            '_question': state['question'],
            '_answer': state['question'].upper(),
            '_notes': 'some notes'
        }

    def node_2(state: OverallState) -> OutputState:
        print("--- Node 2 ---")
        return {'answer': state['_answer']}

    builder = StateGraph(OverallState, input=InputState, output=OutputState)
    builder.add_node('node_1', node_1)
    builder.add_node('node_2', node_2)
    builder.add_edge(START, 'node_1')
    builder.add_edge('node_1', 'node_2')
    builder.add_edge('node_2', END)

    graph = builder.compile()

    state = graph.invoke({'question': 'Some Question'})
    print(state['answer'])
