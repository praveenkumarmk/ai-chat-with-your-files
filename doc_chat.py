from pathlib import Path

import gradio as gr

from ask_llm_with_query import ask
from chunk_buildIndex import ingest_file

state = {"db": None}                    # ① remember the index across turns

def upload(files):
    for file in files:
        doc_id = Path(file.name).stem
        state["db"] = ingest_file(file.name, doc_id, "tenant-acme")   # ② re-index whenever a new file arrives
    names = ", ".join(Path(f.name).name for f in files)
    return f"✅ Indexed: {names}. Ask me anything about them."

def chat(message, history):
    if state["db"] is None:
        return "Please upload a document first 📄"
    return ask(state["db"], message)

with gr.Blocks(title="📄 Chat with your DOC") as demo:
    gr.Markdown("## 📄 Chat with your documents")
    files  = gr.File(label="Upload PDF, DOCX, or HTML files", file_types=[".pdf", ".docx", ".html", ".htm"], file_count="multiple")
    status = gr.Markdown()
    files.upload(upload, inputs=files, outputs=status)
    gr.ChatInterface(fn=chat)

demo.launch(share=True)   