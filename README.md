# Simple LLM Server

A minimal FastAPI service that serves a small open-source instruction-tuned
LLM ([Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct))
for text generation, packaged as a Docker image so it runs the same way
anywhere.

## Project layout

```
project/
├── app/
│   ├── __init__.py
│   ├── model.py      # loads the model + tokenizer, runs generation
│   └── main.py        # FastAPI app: HTTP endpoints
├── requirements.txt    # Python dependencies (pinned versions)
├── Dockerfile           # how to build the container image
├── .dockerignore
└── README.md
```

## Running it

```bash
docker build -t simple-llm-server .
docker run -p 8000:8000 simple-llm-server
```

Then, in another terminal:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain Docker in one sentence."}'
```

Or open `http://localhost:8000/docs` in a browser for FastAPI's
auto-generated interactive API docs (Swagger UI) — you can try requests
directly from there without curl.

## What's actually happening behind the scenes

### 1. The model

`Qwen/Qwen2.5-0.5B-Instruct` is a pretrained, instruction-tuned LLM published
on Hugging Face Hub — nobody in this project trained it. "Instruction-tuned"
means it was fine-tuned to follow chat-style prompts ("Explain X", "Write a
Y") rather than just complete arbitrary text. It's small (0.5 billion
parameters) specifically so it runs at usable speed on a CPU with no GPU;
larger models (7B+) would be far slower or need a GPU.

`app/model.py` uses the Hugging Face `transformers` library to:
- **Tokenizer**: converts text into integer token IDs the model understands,
  and back again. `apply_chat_template` wraps your raw prompt in the
  special tokens Qwen expects (e.g. marking "this is a user turn") — every
  chat model has its own template, and using the wrong one degrades output
  quality.
- **Model**: `AutoModelForCausalLM` loads the actual neural network weights
  — a causal (autoregressive) language model that predicts the next token
  given all previous ones, repeatedly, until it produces an end-of-text
  token or hits `max_new_tokens`.
- **`model.generate(...)`**: runs that repeated-prediction loop. `do_sample`,
  `temperature`, and `top_p` control randomness — higher temperature = more
  varied/creative, lower = more deterministic/repetitive.

The model is loaded **once**, into a single shared `ChatModel` instance, when
the server process starts — not on every request. Loading takes a couple of
seconds and holds the weights in memory (~2GB of RAM for this model in
float32); reloading per-request would make every call painfully slow.

### 2. The FastAPI app (`app/main.py`)

- `@asynccontextmanager lifespan(app)` is FastAPI's startup/shutdown hook.
  `model.load()` runs here, once, before the server starts accepting
  requests — this is the "load once" behavior above, wired into the
  framework.
- `GenerateRequest` / `GenerateResponse` are Pydantic models. FastAPI uses
  them to automatically: validate incoming JSON (reject bad requests with a
  clear 422 error before your code even runs), parse it into a typed Python
  object, and serialize your return value back to JSON. This also powers
  the auto-generated `/docs` page.
- `POST /generate` is the main endpoint: takes a prompt, calls
  `model.generate(...)`, returns the text.
- `GET /health` is a cheap endpoint that just reports the process is up and
  the model is loaded. This is a standard production pattern — load
  balancers, Kubernetes, or Docker itself can poll `/health` to know whether
  an instance is actually ready to serve traffic before routing requests to
  it (not wired into an actual healthcheck yet — see "Ideas for later").
- Uvicorn is the ASGI server that actually listens on a TCP port and hands
  requests to your FastAPI app — FastAPI itself is just the framework/router,
  it doesn't listen on sockets on its own.

### 3. The Docker image

Docker packages your app, the exact Python version, and every dependency
into one self-contained image, so "works on my machine" becomes "works
anywhere Docker runs" — the same image runs identically on your laptop, a
teammate's machine, or a cloud server.

Reading the `Dockerfile` top to bottom, each `RUN`/`COPY` is a **layer**,
cached independently by Docker:

1. `FROM python:3.11-slim` — start from a minimal official Python image
   (not full Ubuntu) to keep the image smaller.
2. Install CPU-only PyTorch **first, separately**, from PyTorch's own package
   index. The default PyPI `torch` build bundles NVIDIA CUDA libraries for
   GPU use — several GB you don't need on a CPU-only machine. Installing
   this in its own layer, before copying app code, also means Docker can
   reuse this slow step from cache on rebuilds as long as the torch version
   doesn't change.
3. Copy `requirements.txt` and install the rest — again *before* copying
   the app code, for the same caching reason: your app code changes far more
   often than your dependencies, so this ordering means most rebuilds only
   re-run the fast "copy code" step, not the slow "reinstall everything" step.
4. Copy the `app/` source code in.
5. `RUN python -c "from app.model import model; model.load()"` — this is the
   interesting one: it runs the model-loading code **during the image
   build**, which downloads the ~1GB of model weights from Hugging Face Hub
   and caches them inside the image filesystem. The trade-off:
   - ✅ The resulting container needs **no network access at runtime** and
     starts in seconds (just reading local files), which is what you want
     in production — no dependency on Hugging Face Hub being reachable.
   - ❌ The image itself is large (~2GB+) and `docker build` is slow the
     first time (downloads happen at build time instead of run time).
   - The alternative (download at container *startup* instead, e.g. via a
     mounted volume cache) keeps the image small but means every fresh
     container needs internet access and a slower first request. Fine to
     switch to later if image size becomes a problem.
6. `EXPOSE 8000` documents which port the app listens on (doesn't actually
   publish it — that's what `-p 8000:8000` on `docker run` does, mapping
   container port 8000 to your machine's port 8000).
7. `CMD [...]` — the command that runs when a container starts: launch
   Uvicorn serving the FastAPI app.

### Request flow, end to end

1. `docker run -p 8000:8000 simple-llm-server` starts a container from the
   image; Docker maps your machine's port 8000 to the container's port 8000.
2. Inside the container, Uvicorn starts, FastAPI's `lifespan` hook runs
   `model.load()` (fast — reads the weights already baked into the image),
   then Uvicorn starts accepting connections.
3. `curl -X POST .../generate -d '{"prompt": "..."}'` sends an HTTP request
   from your machine, through the port mapping, into the container.
4. Uvicorn receives it and hands it to FastAPI's router, which matches
   `POST /generate` and validates the JSON body against `GenerateRequest`.
5. Your `generate()` function runs, calling the shared `model` instance,
   which tokenizes the prompt, runs the neural network's generation loop,
   and decodes the output tokens back to text.
6. FastAPI serializes the return value to JSON and sends the HTTP response
   back out through the same path.

## Ideas for later (not built yet, on purpose — keeping v1 simple)

- Streaming responses (token-by-token, like ChatGPT's UI) instead of waiting
  for the full answer.
- Docker `HEALTHCHECK` wired to the `/health` endpoint.
- Concurrency: right now requests are serialized with a lock; a queue or
  multiple worker processes would let requests overlap.
- Quantization (e.g. via `bitsandbytes` or GGUF) to shrink memory/CPU use
  further — this is exactly what the course covers.
- Swapping in a GPU base image + `nvidia/cuda` runtime if you get access to
  a GPU machine.
- Authentication/rate limiting before exposing this beyond localhost.
