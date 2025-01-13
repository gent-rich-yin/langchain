import gradio as gr

if __name__ == '__main__':
    with gr.Blocks(fill_height=True) as demo:
        gr.Chatbot(scale=1)
        gr.Textbox(scale=0)

    demo.launch()