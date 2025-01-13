import gradio as gr

if __name__ == '__main__':
    with gr.Blocks() as demo:
        with gr.Row(equal_height=True):
            textbox = gr.Textbox()
            btn2 = gr.Button("Button 2")

    demo.launch()