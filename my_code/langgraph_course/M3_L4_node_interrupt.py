from dotenv import load_dotenv
from langgraph.errors import NodeInterrupt

from langgraph.prebuilt.tool_node import tools_condition, ToolNode

from langgraph.checkpoint.memory import MemorySaver

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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

    sys_msg = """
    You are an assistant that can perform arithmetic operations. 
    Please always use the provided tools.
    Do not calculate based on you existing experience and knowledge.
    Do not do error checking yourself.
    """

    system_message = SystemMessage(content=sys_msg)

    def assistant(state: MessagesState):
        return {'messages': [model_with_tools.invoke([system_message] + state['messages'])]}

    def validate_tool_call(state: MessagesState):
        for m in state['messages']:
            if isinstance(m, AIMessage):
                for tool_call in m.tool_calls:
                    if tool_call['name'] == 'divide' and tool_call['args']['b'] == 0:
                        raise NodeInterrupt('Dividend must not be zero')

    builder = StateGraph(MessagesState)
    builder.add_node('assistant', assistant)
    builder.add_node('validate_tool_call', validate_tool_call)
    builder.add_node('tools', ToolNode(tools))
    builder.add_edge(START, 'assistant')
    builder.add_edge('assistant', 'validate_tool_call')
    builder.add_conditional_edges('validate_tool_call', tools_condition)
    builder.add_edge('tools', 'assistant')

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    # this will go without interrupt
    # input_message = {'messages': [HumanMessage(content='multiple 2 and 3')]}
    # thread = {'configurable': {'thread_id': '1'}}
    # for event in graph.stream(input_message, thread, stream_mode='values'):
    #     event['messages'][-1].pretty_print()

    input_message = {'messages': [HumanMessage(content='divide 6 by 0')]}
    thread = {'configurable': {'thread_id': '2'}}
    for event in graph.stream(input_message, thread, stream_mode='values'):
        event['messages'][-1].pretty_print()

    state = graph.get_state(thread)
    print(state.values['messages'])
    state.values['messages'][-1].content[1]['input']['b'] = 2
    state.values['messages'][-1].tool_calls[0]['args']['b'] = 2
    graph.update_state(thread, state.values, as_node='validate_tool_call')
    state = graph.get_state(thread)
    print(state.values['messages'])
    for event in graph.stream(None, thread, stream_mode='values'):
        event['messages'][-1].pretty_print()

