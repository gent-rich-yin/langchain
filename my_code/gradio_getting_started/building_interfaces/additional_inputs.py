import gradio as gr

def generate_fake_image(prompt, seed, initial_image=None):
    return f"Used seed: {seed}", "https://dummyimage.com/300/09f.png"

if __name__ == '__main__':
    demo = gr.Interface(
        generate_fake_image,
        inputs=[gr.Textbox(label='Prompt')],
        additional_inputs=[
            gr.Slider(0, 1000, label='Seed'),
            gr.Image(label='Initial image')
        ],
        outputs=[gr.Textbox(label='Chosen seed'), gr.Image(label='Fake image')]
    )
    demo.launch()
