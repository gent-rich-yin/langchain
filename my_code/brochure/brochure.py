from typing import TypedDict, Annotated, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from chatbot.runnable_with_tools import RunnableWithTools
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


class WebPage(BaseModel):
    """ Information about a web page, including its title, content and useful links """
    url: str = Field(description="URL of this web page")
    title: Optional[str] = Field(default=None, description="The web page title")
    content: Optional[str] = Field(default=None, description="Web page content")
    links: list[str] = Field(default_factory=list, description="the useful links on this web page")

@tool
def get_web_page(url: Annotated[str, 'the url of the web page']) -> Annotated[WebPage, 'information on the web page']:
    """ Retrieve information about a web page, including its title, content and useful links """
    print(f'Retrieving {url}')
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title.string if soup.title else None
    content = None
    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        content = soup.body.get_text(separator="\n", strip=True)
    links = [link.get('href') for link in soup.find_all('a') if link.get('href') is not None]
    return WebPage(url=url, title=title, content=content, links=links)

TOOLS = {
    "get_web_page": get_web_page
}

def run(model: RunnableWithTools, company_name: str, company_web_site: str) -> str:
    system = """
    You are an assistant that can crate company brochure from company's web site.
    You can do this in a few steps:
    1. Retrieve web page information from the provide company web site url
    2. Find out the links that are useful for making company brochure
    3. Retrieve web page information from these links
    4. Create a company brochure from information on these wwn pages 
    """

    user = """
    Please create a company brochure for {company_name}. Its web site url is {company_web_site}
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system),
        ('user', user)
    ])
    prompt = prompt_template.invoke({'company_name': company_name, 'company_web_site': company_web_site})
    return model.invoke(prompt.to_messages(), max_depth=20).content


if __name__ == '__main__':
    load_dotenv()
    # llm = ChatOpenAI(temperature=0, model_name='gpt-4o-mini')
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
    # llm = ChatOllama(model = "qwen2.5", temperature = 0, num_thread=8)
    model_with_tools = RunnableWithTools(bound=model.bind_tools([get_web_page]), tools=TOOLS)

    brochure = run(model_with_tools, "Mizuho Americas", "https://www.mizuhogroup.com/americas")
    print(brochure)
