import gradio as gr

if __name__ == '__main__':
    with gr.Blocks() as demo:
        name = gr.Textbox(label="Name")
        output = gr.Textbox(label="Output Box")
        greet_btn = gr.Button("Greet")
        trigger = gr.Textbox(label="Trigger Box")

        def greet(name, evt_data: gr.EventData) -> (str, str):
            return "Hello " + name + "!", evt_data.target.__class__.__name__

        def clear_name() -> str:
            return ""

        gr.on(
            triggers=[name.submit, greet_btn.click],
            fn=greet,
            inputs=name,
            outputs=[output, trigger],
        ).then(clear_name, outputs=[name])

    demo.launch()
