import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from main import app
import gradio as gr

# Mount FastAPI app onto Gradio UI interface
with gr.Blocks(title="Chroma Rerank RAG API") as demo:
    gr.Markdown("# ⚡ Chroma Rerank RAG Backend API is Live!")
    gr.Markdown("FastAPI dual-stage RAG endpoints `/api/v1/upload` and `/api/v1/query` are active.")

app = gr.mount_gradio_app(app, demo, path="/")
