import asyncio

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import RemoveMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.constants import START, END

from langgraph.graph import StateGraph, MessagesState

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    class State(MessagesState):
        summary: str

    def call_model(state: State) -> State:
        summary = state.get('summary', '')
        messages = state['messages']
        if summary:
            system_message = f'Summary of conversation earlier: {summary}'
            messages = [SystemMessage(content=system_message)] + messages
        response = model.invoke(messages)
        return {'messages': [response], 'summary': summary}

    def summarize_conversation(state: State) -> State:
        summary = state.get('summary', '')
        if summary:
            summary_message = (
                f'This is summary of the conversation to date: {summary}\n\n'
                'Extend the summary by taking into account the new messages above:'
            )
        else:
            summary_message = 'Create a summary of the conversation above:'

        messages = state['messages'] + [HumanMessage(content=summary_message)]
        response = StrOutputParser().invoke(model.invoke(messages))

        delete_messages = [RemoveMessage(m.id) for m in state['messages'][:-2]]
        return {'messages': delete_messages, 'summary': response}

    def should_continue(state: State) -> str:
        return 'summarize_conversation' if len(state['messages']) > 6 else END

    builder = StateGraph(State)
    builder.add_node('conversation', call_model)
    builder.add_node(summarize_conversation)
    builder.add_edge(START, 'conversation')
    builder.add_conditional_edges('conversation', should_continue)
    builder.add_edge('summarize_conversation', END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    config = {'configurable': {'thread_id': '1'}}
    input_message = HumanMessage(content="hi! I'm Lance")
    for chunk in graph.stream({'messages': [input_message], 'summary': ''}, config, stream_mode='updates'):
        print(f'Message count: {len(chunk['conversation']['messages'])}')
        print(f'Last message: {chunk['conversation']['messages'][-1].content}')

    config = {'configurable': {'thread_id': '2'}}
    input_message = HumanMessage(content="hi! I'm Lance")
    for chunk in graph.stream({'messages': [input_message], 'summary': ''}, config, stream_mode='values'):
        print(f'Message count: {len(chunk['messages'])}')
        print(f'Last message: {chunk['messages'][-1].content}')

    async def demo_async_1() -> None:
        config = {'configurable': {'thread_id': '3'}}
        input_message = HumanMessage(content="Tell me about 76ers")
        async for event in graph.astream_events({'messages': [input_message], 'summary': ''}, config, version='v2'):
            print(f'Node: {event['metadata'].get('langgraph_node', '')}. Type: {event['event']}. Name: {event['name']}')
            # print(f'Last message: {chunk['messages'][-1].content}')
    asyncio.run(demo_async_1())

    async def demo_async_2() -> None:
        config = {'configurable': {'thread_id': '4'}}
        input_message = HumanMessage(content="Tell me about 76ers")
        async for event in graph.astream_events({'messages': [input_message], 'summary': ''}, config, version='v2'):
            if event['event'] == 'on_chat_model_stream' and event['metadata'].get('langgraph_node', '') == 'conversation':
                print(event['data']['chunk'].content, end='|')
    asyncio.run(demo_async_2())
