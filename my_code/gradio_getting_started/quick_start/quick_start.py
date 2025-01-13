import gradio as gr

def greet(name: str) -> str:
    return f'Hello {name}'

if __name__ == '__main__':
    demo = gr.Interface(greet, [gr.Textbox(label='Name')], [gr.Textbox('Greeting')])
    demo.launch(share=True)
