from data_load import load_interest_rate_curve, supported_currencies
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent

import gradio as gr

if __name__ == '__main__':
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    tools = [load_interest_rate_curve]

    # parameter: supported_currencies
    prompt_template = """
    system
    You are an agent named RichBot. You are designed to load and analyze interest rates 
    in {supported_currencies} for any trading day.
    Given an input question, you can invoke the tool to load interest rates for the given date and currency,
    then analyze the loaded interest rates to answer the questions.
    If the input question is not one the supported currency, just answer "I don't know"
    If the tool doesn't return meaningful interest rate curve object, you can also answer "I don't know".
    You can invoke the tool as many times as you like to load relevant interest rates.
    Please only answer the question based on the data returned from the tools.
    Don't fake the data. Don't answer based on your past knowledge.   
    """
    system_message = prompt_template.format(supported_currencies=supported_currencies)

    agent_executor = create_react_agent(model, tools, state_modifier=system_message)

    # for step in agent_executor.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="values"):
    #     step["messages"][-1].pretty_print()

    def ir_curve(question: str, history: list[dict]) -> str:
        # answer = agent_executor.invoke({"messages": [{"role": "user", "content": question}]})
        # return answer['messages'][-1].content
        messages = history + [{"role": "user", "content": question}]
        for step in agent_executor.stream({"messages": messages}, stream_mode="values"):
            last_message = step["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.content:
                if isinstance(last_message.content, list):
                    for c in last_message.content:
                        if 'text' in c:
                            yield c['text']
                elif isinstance(last_message.content, str):
                    yield last_message.content

    gr.ChatInterface(
        fn=ir_curve,
        type="messages",
        title="RichBot",
        description=f"""
        A simple bot that can load and analyze interest rates in these currencies: {supported_currencies}. <br/>
        Data comes from https://rfr.ihsmarkit.com
        """,
        theme='John6666/YntecDark'
    ).launch(share=True)