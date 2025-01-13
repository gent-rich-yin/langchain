import gradio as gr
from jedi.inference.gradual.typing import TypedDict

scores: list[int] = []

class Output(TypedDict):
    current_message: str
    history: list[str]

def store_messages(message: str, history: list[str]) -> (Output, list[str]):
    output = {
        'current_message': message,
        'history': history[::]
    }
    history.insert(0, message)
    return output, history

if __name__ == '__main__':
    demo = gr.Interface(store_messages, [gr.Textbox(label='Message'), gr.State(value=[])], [gr.JSON(label='State'), gr.State()])
    demo.launch(share=True)
