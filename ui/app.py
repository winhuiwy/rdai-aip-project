"""Gradio chat UI. A separate small service from the FastAPI backend — it
holds no model itself, it just renders a chat window and forwards every
message (plus history) to the backend's /chat endpoint over HTTP.
"""

import os

# Must be set before `import gradio` — avoids gradio phoning home to its
# analytics endpoint on startup, which isn't needed and can hang/fail on
# a restricted network.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def chat_fn(message: str, history: list[dict]) -> str:
    # `history` is already in [{"role": ..., "content": ...}, ...] form
    # (type="messages" below) — just add the new user turn and forward it.
    messages = history + [{"role": "user", "content": message}]
    resp = requests.post(f"{BACKEND_URL}/chat", json={"messages": messages}, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


demo = gr.ChatInterface(
    fn=chat_fn,
    type="messages",
    title="Simple LLM Chat",
    description=f"Talking to the model via {BACKEND_URL}/chat",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
