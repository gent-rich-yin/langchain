import sqlite3

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import RemoveMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import START, END

from langgraph.graph import StateGraph, MessagesState

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
    # In memory
    # conn = sqlite3.connect(':memory:')
    # Saved to local db
    db_path = 'example.db'
    conn = sqlite3.connect(db_path, check_same_thread=False)

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

        delete_messages = [RemoveMessage(m.id) for m in state['messages']]
        return {'messages': delete_messages, 'summary': response}

    def should_continue(state: State) -> str:
        if len(state['messages']) > 6:
            return 'summarize_conversation'
        else:
            return END

    builder = StateGraph(State)
    builder.add_node('conversation', call_model)
    builder.add_node(summarize_conversation)
    builder.add_edge(START, 'conversation')
    builder.add_conditional_edges('conversation', should_continue)
    builder.add_edge('summarize_conversation', END)

    memory = SqliteSaver(conn)
    graph = builder.compile(checkpointer=memory)

    config = {'configurable': {'thread_id': '1'}}

    saved_state = graph.get_state(config)
    print(saved_state.values['summary'])

    input_message = HumanMessage(content="Can you craft a greeting from me to James Harden? Please mention my name in the greeting.")
    output = graph.invoke({'messages': [input_message]}, config)
    for m in output['messages'][-1:]:
        m.pretty_print()

    print(output['summary'])

