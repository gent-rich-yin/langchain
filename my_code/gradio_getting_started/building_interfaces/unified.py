import gradio as gr

def generate_text(text_prompt: str) -> str:
  return text_prompt.upper()

if __name__ == '__main__':
    textbox = gr.Textbox()
    demo = gr.Interface(generate_text, textbox, textbox, live=True)
    demo.launch()