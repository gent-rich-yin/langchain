from dotenv import load_dotenv

from chatbot.Position import BookPositions
from chatbot.my_tools import get_positions
from chatbot.runnable_with_tools import RunnableWithTools
from langchain_anthropic import ChatAnthropic
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# from langchain_core.output_parsers.string import StrOutputParser
from langchain_ollama import ChatOllama

from langchain_openai import ChatOpenAI

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

PROMPT = "> "
TOOLS = {
    "get_positions": get_positions
}

def run():
    load_dotenv()
    # llm = ChatOpenAI(temperature=0, model_name='gpt-4o-mini')
    llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
    # llm = ChatOllama(model = "qwen2.5", temperature = 0, num_thread=8)
    llm_with_tools = RunnableWithTools(bound=llm.bind_tools([get_positions]), tools=TOOLS)
    history = [SystemMessage(content='You are a Fixed Income market support personnel. Your name is Andy')]
    print("-- ChatBox --")
    print("Enter /q to quit")
    while True:
        msg = input(PROMPT)
        if msg == '/q':
            break
        human_message = HumanMessage(content=msg)
        history.append(human_message)
        res = llm_with_tools.invoke(input=history)
        history.append(res)
        if res.tool_calls:
            if isinstance(res.content, list) and res.content[0]['text']:
                print(res.content[0]['text'])
            elif res.content:
                print(res.content)
            for tool_call in res.tool_calls:
                selected_tool = {"get_positions": get_positions}[tool_call["name"].lower()]
                tool_msg = selected_tool.invoke(tool_call)
                if isinstance(tool_msg, BookPositions):
                    history.append(ToolMessage(content=tool_msg.model_dump_json(), tool_call_id=tool_call["id"], status='success'))
                    print(tool_call["name"])
                    print(tool_msg.model_dump_json())
                else:
                    history.append(ToolMessage(content=str(tool_msg), tool_call_id=tool_call["id"], status='success'))
                    print(tool_call["name"])
                    print(str(tool_msg))
        else:
            print(res.content)


if __name__ == '__main__':
    run()
