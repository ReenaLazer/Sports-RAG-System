import gradio as gr
from rag_pipeline import ask_question

def chatbot_response(message, history):
    response = ask_question(message)
    return response

custom_css = """
footer {
    display: none !important;
}

/* Remove extra bottom spacing */
.gradio-container {
    min-height: 100vh !important;
    font-family: 'Inter', sans-serif !important;
}

/* Chat container height */
.chatbot {
    height: 70vh !important;
}

/* Top title spacing */
h1 {
    margin-bottom: 5px !important;
}
"""

dark_theme = gr.themes.Soft()

with gr.Blocks(
    theme=dark_theme,
    css=custom_css,
    title="RAG - Chatbot"
) as demo:

    with gr.Row():
        gr.Markdown("# Sports RAG Assistant")

    with gr.Row():
        gr.Markdown("A Sports Chatbot")

    chatbot = gr.ChatInterface(
        fn=chatbot_response,
        chatbot=gr.Chatbot(
            height=500,
            elem_classes="chatbot"
        )
    )

demo.launch(inbrowser=True)