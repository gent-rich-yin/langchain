import asyncio

from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    # sync stream
    chunks = []
    for chunk in model.stream("What color is the sky?"):
        chunks.append(chunk)
        print(chunk.content, end='|', flush=True)
    # chunk is AIMessageChunk, and addable
    full_message = chunks[0]
    for chunk in chunks[1:]:
        full_message += chunk
    print()
    print(full_message.content)
    print()

    # async stream
    async def astream():
        async for chunk in model.astream("What color is the sky?"):
            print(chunk.content, end='|', flush=True)
    asyncio.run(astream())
    print()

    # async chain
    async def async_chain():
        prompt = ChatPromptTemplate.from_template('Tell me a joke about {topic}')
        parser = StrOutputParser()
        chain = prompt | model | parser
        async for chunk in chain.astream({'topic': 'parrot'}):
            print(chunk, end='|', flush=True)
    asyncio.run(async_chain())
    print()

    async def async_json_output():
        chain = model | JsonOutputParser()
        async for text in chain.astream(
                "output a list of the countries france, spain and japan and their populations in JSON format. "
                'Use a dict with an outer key of "countries" which contains a list of countries. '
                "Each country should have the key `name` and `population`"
        ):
            print(text, flush=True)
    asyncio.run(async_json_output())
    print()

