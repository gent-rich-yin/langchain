import operator
from typing import List, Optional, NotRequired

import faiss
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated

from langchain_anthropic import ChatAnthropic
from langgraph.constants import START, END

from langgraph.graph import StateGraph, MessagesState

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    class Log(TypedDict):
        id: str
        question: str
        docs: NotRequired[List]
        answer: str
        grade: NotRequired[int]
        grader: NotRequired[str]
        feedback: NotRequired[str]

    class FailureAnalysisState(TypedDict):
        cleaned_logs: List[Log]
        failures: List[Log]
        fa_summary: str
        processed_logs: List[str]

    class FailureAnalysisOutputState(TypedDict):
        fa_summary: str
        processed_logs: List[str]

    def get_failures(state: FailureAnalysisState):
        return {'failures': [log for log in state['cleaned_logs'] if 'grade' in log]}

    def generate_summary(state: FailureAnalysisState) -> FailureAnalysisOutputState:
        return {
            'fa_summary': 'Poor quality retrieval',
            'processed_logs': [f'failure-analysis-on-log-{log["id"]}' for log in state['failures']]
        }

    fa_builder = StateGraph(input=FailureAnalysisState, output=FailureAnalysisOutputState)
    fa_builder.add_node('get_failures', get_failures)
    fa_builder.add_node('generate_summary', generate_summary)
    fa_builder.add_edge(START, 'get_failures')
    fa_builder.add_edge('get_failures', 'generate_summary')
    fa_builder.add_edge('generate_summary', END)

    class QuestionSummarizationState(TypedDict):
        cleaned_logs: List[Log]
        qs_summary: str
        report: str
        processed_logs: List[str]

    class QuestionSummarizationOutputState(TypedDict):
        report: str
        processed_logs: List[str]

    def generate_summary(state: QuestionSummarizationState):
        return {
            'qs_summary': 'Questions focused on Ollama usage and vector store',
            'processed_logs': [f'failure-analysis-on-log-{log["id"]}' for log in state['cleaned_logs']]
        }

    def send_to_slack(state: QuestionSummarizationState):
        return {'report': 'foo bar'}

    qs_builder = StateGraph(input=QuestionSummarizationState, output=QuestionSummarizationOutputState)
    qs_builder.add_node('generate_summary', generate_summary)
    qs_builder.add_node('send_to_slack', send_to_slack)
    qs_builder.add_edge(START, 'generate_summary')
    qs_builder.add_edge('generate_summary', 'send_to_slack')
    qs_builder.add_edge('send_to_slack', END)

    class EntryState(TypedDict):
        raw_logs: List[Log]
        cleaned_logs: Annotated[list[str], operator.add]
        fa_summary: str
        report: str
        processed_logs: Annotated[list[int], operator.add]

    def clean_logs(state: EntryState):
        return {'cleaned_logs': state['raw_logs']}

    entry_builder = StateGraph(EntryState)
    entry_builder.add_node('clean_logs', clean_logs)
    entry_builder.add_node('question_summarization', qs_builder.compile())
    entry_builder.add_node('failure_analysis', fa_builder.compile())
    entry_builder.add_edge(START, 'clean_logs')
    entry_builder.add_edge('clean_logs', 'question_summarization')
    entry_builder.add_edge('clean_logs', 'failure_analysis')
    entry_builder.add_edge('question_summarization', END)
    entry_builder.add_edge('failure_analysis', END)

    graph = entry_builder.compile()

    question_answer = Log(
        id='1',
        question='How do I import ChatOllama',
        answer='To import Ollama, use "from langchain_community.char_models import ChatOllama'
    )

    question_answer_feedback = Log(
        id='2',
        question='How to use vector store',
        answer='You should do this , this to use vector store',
        grade=0,
        grader='Document Retrieval Recall',
        feedback='Not very useful'
    )

    raw_logs = [question_answer, question_answer_feedback]
    print(graph.invoke({'raw_logs': raw_logs}))


