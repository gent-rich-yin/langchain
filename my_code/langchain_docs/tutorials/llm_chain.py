from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_openai_messages
from langchain_core.prompts import ChatPromptTemplate

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    # invoke with messages
    messages = [
        SystemMessage(content='Translate the following from English into Italian'),
        HumanMessage(content='Hi!')
    ]
    print(model.invoke(messages).content)

    # invoke with Open AI format
    messages = [
        {'role': 'system', 'content': 'Translate the following from English into Italian'},
        {'role': 'user', 'content': 'Hi!'}
    ]
    print(model.invoke(messages).content)

    # this doesn't work: Message dict must contain 'role' and 'content' keys
    # messages = [
    #     {'system': 'Translate the following from English into Italian'},
    #     {'user': 'Hi!'}
    # ]
    # print(model.invoke(messages).content)

    # this works
    messages = [
        ('system', 'Translate the following from English into Italian'),
        ('user', 'Hi!')
    ]
    print(model.invoke(messages).content)

    # convert LangChain Messages to Open AI format
    messages = [
        SystemMessage(content='Translate the following from English into Italian'),
        HumanMessage(content='Hi!')
    ]
    print(model.invoke(convert_to_openai_messages(messages)).content)

    # streaming
    messages = [
        SystemMessage(content='Translate the following from English into Italian'),
        HumanMessage(content='Hi!')
    ]
    for token in model.stream(messages):
        print(token.content, end='|')
    print()

    # prompt template from messages (in strings)
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', 'Translate the following from English into {language}'),
        ('user', '{text}')
    ])
    prompt = prompt_template.invoke({"language": "Italian", "text": "Hi!"})
    print(model.invoke(prompt).content)

    # this doesn't work
    # prompt_template = ChatPromptTemplate.from_messages([
    #     SystemMessage(content='Translate the following from English into {language}'),
    #     HumanMessage(content='{text}')
    # ])
    # prompt = prompt_template.invoke({"language": "Italian", "text": "Hi!"})
    # print(model.invoke(prompt).content)

    # ChatPromptTemplate.from_messages doesn't work with Open AI format as well.
    # @TODO: Any other ways to use ChatPromptTemplate.from_messages?
    # @TODO: Example on ChatPromptTemplate.from_template?
