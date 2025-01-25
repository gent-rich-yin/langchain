from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.prebuilt import create_react_agent

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()

    prompt_template = """
    system
    You are an agent designed to interact with a SQL database.
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
    system_message = prompt_template.format(dialect="SQLite", top_k=5)

    agent_executor = create_react_agent(model, tools, state_modifier=system_message)

    # question = "Which country's customers spent the most?"
    # for step in agent_executor.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="values"):
    #     step["messages"][-1].pretty_print()

    # question = "what is the average spending among all countries? Just tell me the number, no explanation please."
    # answer = agent_executor.invoke({"messages": [{"role": "user", "content": question}]})
    # print(answer['messages'][-1].content)

    question = "Describe the playlisttrack table"
    for step in agent_executor.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="values"):
        step["messages"][-1].pretty_print()