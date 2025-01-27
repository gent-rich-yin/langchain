import random
from typing import TypedDict

from langgraph.constants import START, END

from langgraph.graph import StateGraph


class State(TypedDict):
    graph_state: str

def node_1(state: State) -> State:
    print("--- Node 1 ---")
    return {'graph_state': state['graph_state'] + " I am"}

def node_2(state: State) -> State:
    print("--- Node 2 ---")
    return {'graph_state': state['graph_state'] + " happy!"}

def node_3(state: State) -> State:
    print("--- Node 3 ---")
    return {'graph_state': state['graph_state'] + " sad!"}

def random_selection(state: State) -> str:
    return 'node_2' if random.random() < 0.5 else 'node_3'

if __name__ == '__main__':
    builder = StateGraph(State)
    builder.add_node('node_1', node_1)
    builder.add_node('node_2', node_2)
    builder.add_node('node_3', node_3)

    builder.add_edge(START, 'node_1')
    builder.add_conditional_edges('node_1', random_selection)
    builder.add_edge('node_2', END)
    builder.add_edge('node_3', END)

    graph = builder.compile()

    state = {'graph_state': ''}
    state = graph.invoke(state)
    print(state['graph_state'])
