import sys
import os
import uvicorn

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from main import app as fastapi_app
import gradio as gr

# Build Gradio UI
with gr.Blocks(title="Chroma Rerank RAG API") as demo:
    gr.Markdown("# ⚡ Chroma Rerank RAG Backend API is Live!")
    gr.Markdown("FastAPI dual-stage RAG endpoints `/api/v1/upload` and `/api/v1/query` are active.")

# Mount FastAPI application onto Gradio UI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# Run persistent Uvicorn server loop on port 7860
uvicorn.run(app, host="0.0.0.0", port=7860)
