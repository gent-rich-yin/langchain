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

        @name.change(inputs=name, outputs=greeting)
        def greet(name: str) -> str:
            return f'Hello, {name}'

        @greeting.change(inputs=greeting, outputs=shout)
        def shout(s: str) -> str:
            return s.upper()

    demo.launch(share=True)