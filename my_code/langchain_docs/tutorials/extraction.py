from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic

from typing import Optional
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import tool_example_to_messages


class Person(BaseModel):
    """Information about a person."""
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )

class People(BaseModel):
    """Extracted data about people."""
    people: list[Person]

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            ("human", "{text}"),
        ]
    )

    model_for_person = model.with_structured_output(schema=Person)

    text = "Alan Smith is 6 feet tall and has blond hair."
    prompt = prompt_template.invoke({"text": text})
    print(model_for_person.invoke(prompt))

    model_for_people = model.with_structured_output(schema=People)

    text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
    prompt = prompt_template.invoke({"text": text})
    print(model_for_people.invoke(prompt))

    text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me, and is 1 inch shorter than we."
    prompt = prompt_template.invoke({"text": text})
    print(model_for_people.invoke(prompt))

    # the answer to this one is wrong
    text = "Alan is a dog, is 1 foot tall and has blond fur."
    prompt = prompt_template.invoke({"text": text})
    print(model_for_people.invoke(prompt))

    # reference examples (few-shot prompting)
    messages = [
        {"role": "user", "content": "2 🦜 2"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "2 🦜 3"},
        {"role": "assistant", "content": "5"},
        {"role": "user", "content": "3 🦜 4"},
    ]

    response = model.invoke(messages)
    print(response.content)

    # few-shot prompting with tool_example_to_messages
    examples = [
        (
            "The ocean is vast and blue. It's more than 20,000 feet deep.",
            People(people=[]),
        ),
        (
            "Fiona traveled far from France to Spain.",
            People(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
        ),
        (
            "A dog is not a person",
            People(people=[]),
        ),
    ]

    messages = []

    for txt, tool_call in examples:
        if tool_call.people:
            # This final message is optional for some providers
            ai_response = "Detected people."
        else:
            ai_response = "Detected no people."
        messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))

    for message in messages:
        message.pretty_print()

    message_no_extraction = {
        "role": "user",
        "content": "The solar system is large, but earth has only 1 moon.",
    }

    print(f"Extraction with no example for {message_no_extraction['content']}")
    print(model_for_people.invoke([message_no_extraction]))

    print(f"Extraction with examples for {message_no_extraction['content']}")
    print(model_for_people.invoke(messages + [message_no_extraction]))

    message_no_extraction = {
        "role": "user",
        "content": "Alan is a dog, is 1 foot tall and has blond fur.",
    }
    print(f"Extraction with no example for {message_no_extraction['content']}")
    print(model_for_people.invoke([message_no_extraction]))

    print(f"Extraction with examples for {message_no_extraction['content']}")
    print(model_for_people.invoke(messages + [message_no_extraction]))
