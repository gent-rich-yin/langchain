from dotenv import load_dotenv
from langgraph.prebuilt.tool_node import tools_condition, ToolNode

from langgraph.checkpoint.memory import MemorySaver

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.constants import START

from langgraph.graph import StateGraph, MessagesState

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
    divide 2 numbers and return result
    """
    return a / b

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
    tools = [multiply, add, divide]
    model_with_tools = model.bind_tools(tools)

    system_message = SystemMessage(content='You are an assistant that can perform arithmetic operations')


    def human_feedback(state: MessagesState) -> None:
        """"""

    def assistant(state: MessagesState):
        return {'messages': [model_with_tools.invoke([system_message] + state['messages'])]}

    builder = StateGraph(MessagesState)
    builder.add_node('human_feedback', human_feedback)
    builder.add_node('assistant', assistant)
    builder.add_node('tools', ToolNode(tools))
    builder.add_edge(START, 'human_feedback')
    builder.add_edge('human_feedback', 'assistant')
    builder.add_conditional_edges('assistant', tools_condition)
    builder.add_edge('tools', 'assistant')

    memory = MemorySaver()
    graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

    input_message = {'messages': [HumanMessage(content='multiple 2 and 3')]}
    thread = {'configurable': {'thread_id': '1'}}
    for event in graph.stream(input_message, thread, stream_mode='values'):
        event['messages'][-1].pretty_print()

    # append a new message
    # added_input_message = {'messages': [HumanMessage(content='No, actually please multiple 3 by 3')]}
    # graph.update_state(thread, added_input_message)

    # replace message
    state = graph.get_state(thread)
    state.values['messages'][-1] = HumanMessage(content='Please multiply 4 by 3')
    graph.update_state(thread, state.values, as_node='human_feedback')

    for event in graph.stream(None, thread, stream_mode='values'):
        event['messages'][-1].pretty_print()
