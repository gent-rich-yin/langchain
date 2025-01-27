from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent

import gradio as gr

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    db = SQLDatabase.from_uri("postgresql://postgres:Save0404@127.0.0.1/mortals",
                              schema='bowlingleagueexample')
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()

    prompt_template = """
    system
    You are an agent named RichBot. You are designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run, 
    then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, 
    always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables.    
    """
    system_message = prompt_template.format(dialect=db.dialect, top_k=100)

    agent_executor = create_react_agent(model, tools, state_modifier=system_message)

    # for step in agent_executor.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="values"):
    #     step["messages"][-1].pretty_print()

    def bowling(question: str, history: list[dict]) -> str:
        # answer = agent_executor.invoke({"messages": [{"role": "user", "content": question}]})
        # return answer['messages'][-1].content
        messages = history + [{"role": "user", "content": question}]
        for step in agent_executor.stream({"messages": messages}, stream_mode="values"):
            last_message = step["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.content:
                if isinstance(last_message.content, list):
                    for c in last_message.content:
                        if 'text' in c:
                            yield c['text']
                elif isinstance(last_message.content, str):
                    yield last_message.content

    gr.ChatInterface(
        fn=bowling,
        type="messages",
        title="RichBot",
        description="A simple bot that can answer questions on Bowling League by querying Bowling League database.",
        theme='allenai/gradio-theme'
    ).launch(share=True)