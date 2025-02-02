from langchain_core.messages import RemoveMessage, AIMessage, HumanMessage
from langgraph.constants import START, END

from langgraph.graph import StateGraph, MessagesState

if __name__ == '__main__':
    def filter(state: MessagesState) -> MessagesState:
        print("--- Node 1 ---")
        return {'messages': [RemoveMessage(id=m.id) for m in state['messages'][:-2]]}

    def process(state: MessagesState) -> MessagesState:
        print("--- Node 2 ---")
        return {'messages': [AIMessage(content='This is a processed message.')]}

    builder = StateGraph(MessagesState)
    builder.add_node('filter', filter)
    builder.add_node('process', process)
    builder.add_edge(START, 'filter')
    builder.add_edge('filter', 'process')
    builder.add_edge('process', END)

    graph = builder.compile()

    history = [
        AIMessage(content='Hi, I am your bot', id=1),
        HumanMessage(content='My name is Richard', id=2),
        AIMessage(content='Hi, I can answer questions about financial market', id=3),
        HumanMessage(content='What is the price of Apple stock?', id=4)
    ]

    state = graph.invoke({'messages': history})
    print(state['messages'])
