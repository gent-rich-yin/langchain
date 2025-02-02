import operator
from typing import List

from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START, END

from langgraph.graph import StateGraph, MessagesState

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-latest", temperature=0)

    def str_overwrite(left: str, right: str) -> str:
        return right

    class State(TypedDict):
        question: Annotated[str, str_overwrite]
        answer: Annotated[str, str_overwrite]
        documents: Annotated[str, operator.add]

    def search_web(state: State) -> State:
        """ Retrieve docs from web search """
        tavily_search = TavilySearchResults(max_results=3)
        search_docs = tavily_search.invoke(state['question'])
        formatted_search_docs = "\n\n---\n\n".join([
            f'<Document href="{doc['url']}">\n{doc['content']}\n</Document>'
            for doc in search_docs
        ])
        return {
            'question': state['question'],
            'answer': state['answer'],
            'documents': formatted_search_docs
        }

    def search_wiki(state: State) -> State:
        """ Retrieve docs from wikipedia search """
        search_docs = WikipediaLoader(state['question'], load_max_docs=2).load()
        formatted_search_docs = "\n\n---\n\n".join([
            f'<Document source="{doc.metadata['source']}" page="{doc.metadata['page']}">\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
        return {
            'question': state['question'],
            'answer': state['answer'],
            'documents': formatted_search_docs
        }

    def generate_answer(state: State) -> State:
        """ Generate final answer based on web search and wiki search results """
        template = ChatPromptTemplate.from_template('answer the question:\n\n{question}\n\nwith this context:\n\n{context}')
        chain = template | model | StrOutputParser()
        answer = chain.invoke({'question': state['question'], 'context': state['documents']})
        return {
            'question': state['question'],
            'answer': answer,
            'documents': state['documents']
        }

    builder = StateGraph(State)
    builder.add_node('search_web', search_web)
    builder.add_node('search_wiki', search_wiki)
    builder.add_node('generate_answer', generate_answer)
    builder.add_edge(START, 'search_web')
    builder.add_edge(START, 'search_wiki')
    builder.add_edge('search_web', 'generate_answer')
    builder.add_edge('search_wiki', 'generate_answer')
    builder.add_edge('generate_answer', END)

    graph = builder.compile()
    init_state = {
        'question': 'What is the expectation of Nivdia revenue in 2024',
        'answer': '',
        'documents': ''
    }
    print(graph.invoke(init_state)['answer'])

