import gradio as gr

if __name__ == '__main__':
    import gradio as gr

    with gr.Blocks() as demo:
        with gr.Row():
            text1 = gr.Textbox(label="t1")
            slider2 = gr.Textbox(label="s2")
            drop3 = gr.Dropdown(["a", "b", "c"], label="d3")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                text1 = gr.Textbox(label="prompt 1")
                text2 = gr.Textbox(label="prompt 2")
                inbtw = gr.Button("Between")
                text4 = gr.Textbox(label="prompt 1")
                text5 = gr.Textbox(label="prompt 2")
            with gr.Column(scale=2, min_width=300):
                img1 = gr.Image("https://letsenhance.io/static/8f5e523ee6b2479e26ecc91b9c25261e/1015f/MainAfter.jpg")
                btn = gr.Button("Go")

    demo.launch()