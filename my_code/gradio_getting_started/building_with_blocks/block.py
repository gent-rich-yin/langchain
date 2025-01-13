import gradio as gr

def greet(name: str) -> str:
    return f'Hello, {name}'

if __name__ == '__main__':
    with gr.Blocks() as demo:
        input = gr.Textbox(label='Name', value='Richard')
        output = gr.Textbox(label='Greeting')
        greet_button = gr.Button('Greet')
        greet_button.click(greet, input, output, api_name='Greet')
    demo.launch(share=True)