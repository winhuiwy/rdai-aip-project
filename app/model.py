"""Loads and runs the language model.

Kept separate from main.py so the same loading code can be called both at
Docker build time (to bake the weights into the image) and at container
startup (to load them from the local cache into memory).
"""

import logging
from threading import Lock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

logger = logging.getLogger("app.model")


class ChatModel:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        # transformers' generate() is not guaranteed thread-safe on a single
        # model instance, so we serialize requests with a lock. Fine for a
        # first version; a queue or multiple workers would replace this later.
        self._lock = Lock()

    def load(self) -> None:
        logger.info("Loading model %s ...", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )
        self.model.eval()
        logger.info("Model loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model has not been loaded yet")

        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(chat_text, return_tensors="pt")

        with self._lock, torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Slice off the prompt tokens so we only decode the newly generated ones.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# A single shared instance, loaded once and reused for every request.
model = ChatModel()
