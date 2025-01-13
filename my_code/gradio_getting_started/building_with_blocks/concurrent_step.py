import gradio as gr

if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Hello World!
            Start typing below to see the output.
            """)
        name = gr.Textbox(label='Name', placeholder='What is your name?')
        greeting = gr.Textbox(label='Greeting')
        shout = gr.Textbox(label='Shout')

        @name.change(inputs=name, outputs=[greeting, shout])
        def greet_and_shout(name: str) -> (str, str):
            message = f'Hello, {name}'
            return message, message.upper()

    demo.launch(share=True)