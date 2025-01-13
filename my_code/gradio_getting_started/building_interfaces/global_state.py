import gradio as gr

scores: list[int] = []

def top_3(score: int) -> list[int]:
    scores.append(score)
    return sorted(scores, reverse=True)[:3]

if __name__ == '__main__':
    demo = gr.Interface(top_3, [gr.Number(label='Score')], [gr.JSON(label='Top scores')])
    demo.launch(share=True)
