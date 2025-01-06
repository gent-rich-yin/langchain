from typing import Optional, TypedDict, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    # Pydantic
    class Joke(BaseModel):
        """Joke to tell user."""
        setup: str = Field(description="The setup of the joke")
        punchline: str = Field(description="The punchline to the joke")
        rating: Optional[int] = Field(
            default=None, description="How funny the joke is, from 1 to 10"
        )

    structured_model = model.with_structured_output(Joke)
    joke = structured_model.invoke("Tell me a joke about cats")
    print(joke.model_dump_json())

    # TypedDict
    class Joke(TypedDict):
        """Joke to tell user."""
        setup: Annotated[str, ..., "The setup of the joke"]

        # Alternatively, we could have specified setup as:

        # setup: str                    # no default, no description
        # setup: Annotated[str, ...]    # no default, no description
        # setup: Annotated[str, "foo"]  # default, no description

        punchline: Annotated[str, ..., "The punchline of the joke"]
        rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

    structured_model = model.with_structured_output(Joke)
    joke = structured_model.invoke("Tell me a joke about cats")
    print(joke)

    json_schema = {
        "title": "joke",
        "description": "Joke to tell user.",
        "type": "object",
        "properties": {
            "setup": {
                "type": "string",
                "description": "The setup of the joke",
            },
            "punchline": {
                "type": "string",
                "description": "The punchline to the joke",
            },
            "rating": {
                "type": "integer",
                "description": "How funny the joke is, from 1 to 10",
                "default": None,
            },
        },
        "required": ["setup", "punchline"],
    }
    structured_model = model.with_structured_output(json_schema)
    structured_model.invoke("Tell me a joke about cats")

