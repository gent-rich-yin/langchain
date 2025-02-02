import operator
from typing import List

from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

from langchain_anthropic import ChatAnthropic
from langgraph.constants import START, END

from langgraph.graph import StateGraph, MessagesState

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    def sorting_reducer(left: str|List[str], right: str|List[str]) -> List[str]:
        if isinstance(left, str):
            left = [left]
        if isinstance(right, str):
            right = [right]
        return sorted(left + right)

    class State(TypedDict):
        value: Annotated[List[str], sorting_reducer]

    class ReturnNodeValue:
        def __init__(self, value: str):
            self.value = value

        def __call__(self, state: State) -> State:
            return {'value': [self.value]}

    builder = StateGraph(State)
    builder.add_node('a', ReturnNodeValue('A'))
    builder.add_node('b', ReturnNodeValue('C'))
    builder.add_node('c', ReturnNodeValue('B'))
    builder.add_node('d', ReturnNodeValue('D'))
    builder.add_edge(START, 'a')
    builder.add_edge('a', 'b')
    builder.add_edge('a', 'c')
    builder.add_edge('b', 'd')
    builder.add_edge('c', 'd')
    builder.add_edge('d', END)

    graph = builder.compile()
    print(graph.invoke({'value': []})['value'])

