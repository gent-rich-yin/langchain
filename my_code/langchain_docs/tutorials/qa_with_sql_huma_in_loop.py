from typing import Annotated, NotRequired

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from langgraph.constants import START
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate


class State(TypedDict):
    question: NotRequired[str]
    query: NotRequired[str]
    result: NotRequired[str]
    answer: NotRequired[str]

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    db = SQLDatabase.from_uri("sqlite:///Chinook.db")

    # parameters: dialect, top_k, table_info, input
    query_prompt_template = """
    system
    Given an input question, create a syntactically correct {dialect} query to run to help find the answer. 
    Unless the user specifies in his question a specific number of examples they wish to obtain, 
    always limit your query to at most {top_k} results. 
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
    Pay attention to use only the column names that you can see in the schema description. 
    Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Only use the following tables:
    {table_info}
    Question: {input}
    """
    template = ChatPromptTemplate.from_template(query_prompt_template)

    def write_query(state: State) -> State:
        """Generate SQL query to fetch information."""
        prompt = template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                "input": state["question"],
            }
        )
        structured_llm = model.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}


    def execute_query(state: State) -> State:
        """Execute SQL query."""
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        return {"result": execute_query_tool.invoke(state["query"])}


    def generate_answer(state: State) -> State:
        """Answer question using retrieved information as context."""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f'Question: {state["question"]}\n'
            f'SQL Query: {state["query"]}\n'
            f'SQL Result: {state["result"]}'
        )
        response = model.invoke(prompt)
        return {"answer": str(response.content)}

    graph_builder = StateGraph(State).add_sequence(
        [write_query, execute_query, generate_answer]
    )
    graph_builder.add_edge(START, "write_query")
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])
    config = {"configurable": {"thread_id": "1"}}

    for step in graph.stream({"question": "How many employees are there?"}, config, stream_mode="updates"):
        print(step)

    try:
        user_approval = input("Do you want to go to execute query? (yes/no): ")
    except Exception:
        user_approval = "no"

    if user_approval.lower() == "yes":
        # If approved, continue the graph execution
        for step in graph.stream(None, config, stream_mode="updates"):
            print(step)
    else:
        print("Operation cancelled by user.")