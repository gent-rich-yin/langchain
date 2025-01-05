from typing import Union, Sequence

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def load_pdf_document(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    pages = loader.load()  # every page is a Document
    content = "\n".join([page.page_content for page in pages])
    return content

class PromptData(TypedDict):
    info: str
    question: str

def bot(model: BaseChatModel, data: Union[PromptData, list[PromptData]]) -> Union[str, list[str]]:
    system_message = """
    You are an assistant that can answer questions based on the information retrieved from a PDF file below.
    {info}
    """

    human_message = """
    {question}
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('human', human_message)
    ])

    if isinstance(data, dict):
        prompt = prompt_template.invoke(data)
        return model.invoke(prompt).content
    elif isinstance(data, list):
        prompts = prompt_template.batch(data)
        return [r.content for r in model.batch(prompts)]

if __name__ == '__main__':
    load_dotenv()
    # model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
    model = ChatOpenAI(temperature=0, model_name='gpt-4o-mini')
    # model = ChatOllama(model = "qwen2.5", temperature = 0, num_thread=8)

    file_path = "sec-hot-topics.pdf"
    info = load_pdf_document(file_path)
    print(f'info length: {len(info)}')

    prompts_data = [
        {'info': info, 'question': "What is this course about?"},
        {'info': info, 'question': "What is Regulation S-K?"},
    ]

    answers = bot(model, prompts_data)
    for prompt, answer in zip(prompts_data, answers):
        print(prompt['question'])
        print(answer)
        print()

