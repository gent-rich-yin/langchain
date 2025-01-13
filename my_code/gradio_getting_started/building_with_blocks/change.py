import gradio as gr

if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Hello World!
            Start typing below to see the output.
            """)
        input = gr.Textbox(label='Name', placeholder='What is your name?')
        output = gr.Textbox(label='Greeting')

        @input.change(inputs=input, outputs=output)
        def greet(name: str) -> str:
            return f'Hello, {name}'

    demo.launch(share=True)