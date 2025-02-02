import random
from typing import TypedDict, Literal, NotRequired

from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.constants import START, END

from langgraph.graph import StateGraph


class State(TypedDict):
    name: NotRequired[str]
    mood: NotRequired[str]

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0.7)

    def generate(topic: str) -> str:
        prompt = """
        Please provide a random English {topic}. Just return those words, do not say anything else.
        """
        template = ChatPromptTemplate.from_template(prompt)
        chain = template | model | StrOutputParser()
        return chain.invoke({'topic': topic})

    def node_1(state: State) -> State:
        print("--- Node 1 ---")
        return {'name': generate('person name')}

    def node_2(state: State) -> State:
        print("--- Node 2 ---")
        return {'mood': generate("adjective for describing a person's mood").lower()}

    builder = StateGraph(State)
    builder.add_node('node_1', node_1)
    builder.add_node('node_2', node_2)
    builder.add_edge(START, 'node_1')
    builder.add_edge('node_1', 'node_2')
    builder.add_edge('node_2', END)

    graph = builder.compile()

    state = graph.invoke({'name': 'Richard'})
    print(f'{state['name']} is {state['mood']}')
