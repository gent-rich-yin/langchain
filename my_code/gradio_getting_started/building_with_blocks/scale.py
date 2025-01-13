import gradio as gr

if __name__ == '__main__':
    with gr.Blocks() as demo:
        with gr.Row():
            btn0 = gr.Button("Button 0", scale=0)
            btn1 = gr.Button("Button 1", scale=1)
            btn2 = gr.Button("Button 2", scale=2)

    demo.launch()