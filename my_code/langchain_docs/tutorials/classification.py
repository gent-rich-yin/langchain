from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")

class Classification2(BaseModel):
    sentiment: str = Field(description="The sentiment of the text", enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian", "chinese"]
    )

if __name__ == '__main__':
    load_dotenv()
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    tagging_prompt = ChatPromptTemplate.from_template(
        """
        Extract the desired information from the following passage.
        Only extract the properties mentioned in the 'Classification' function.
        Passage:
        {input}
        """
    )

    model_for_classification = model.with_structured_output(Classification)

    inp = "美方这么做达到目的了吗？事实证明是徒劳的。"
    prompt = tagging_prompt.invoke({"input": inp})
    response = model_for_classification.invoke(prompt)
    print(response.model_dump())

    model_for_classification2 = model.with_structured_output(Classification2)

    inp = "美方这么做达到目的了吗？事实证明是徒劳的。"
    prompt = tagging_prompt.invoke({"input": inp})
    response = model_for_classification2.invoke(prompt)
    print(response.model_dump())

