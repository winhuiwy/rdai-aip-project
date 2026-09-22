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
├── chat.py               # local script for an interactive back-and-forth chat
├── requirements.txt    # Python dependencies (pinned versions)
├── Dockerfile           # how to build the backend's container image
├── .dockerignore
├── ui/                    # Gradio chat UI (separate service, see below)
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml    # runs backend + ui together
└── README.md
```

## Running just the API

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

### Having an actual back-and-forth conversation

`/generate` is single-turn: one prompt in, one reply out, no memory of
anything before it. For a real conversation, use `/chat`, which takes the
*entire* message history and returns the next assistant reply:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "My name is Alex."},
      {"role": "assistant", "content": "Nice to meet you, Alex!"},
      {"role": "user", "content": "What is my name?"}
    ]
  }'
```

Doing this by hand over curl gets tedious fast, so there's also a tiny
interactive script — run it on your host machine (not inside Docker) while
the container is running:

```bash
python chat.py
```

It just loops: read what you type, append it to a local `messages` list,
POST the whole list to `/chat`, print the reply, append that reply too, and
repeat.

## Running the full app (API + web chat UI)

For an actual chat webpage instead of curl/`chat.py`, this project also
ships a small [Gradio](https://www.gradio.app/) UI as a **second service**,
alongside the FastAPI backend. `docker-compose.yml` builds and runs both
together with one command:

```bash
docker compose up --build
```

Then open **http://localhost:7860** in a browser — that's the chat UI.
(The API itself is still directly reachable at http://localhost:8000, same
as before.)

### Why a separate service, and how they talk to each other

`ui/app.py` holds no model and does no inference itself — it's just a
webpage (rendered by Gradio) that, on every message you send, calls the
backend's `POST /chat` over plain HTTP, exactly like `chat.py` does. Keeping
it separate from the backend means:
- The UI container is tiny and builds in seconds (no PyTorch, no model
  weights) — it only needs `gradio` and `requests`.
- Either piece can be redeployed, scaled, or replaced independently (e.g.
  swap this UI for a different frontend later without touching the model
  server at all).

`docker-compose.yml` defines two **services**, `backend` and `ui`, each
built from its own Dockerfile. Compose automatically creates a private
Docker network for them and registers each service's name as a DNS
hostname on that network — so from inside the `ui` container,
`http://backend:8000` reaches the backend container directly, without going
through your machine's `localhost` or the `-p 8000:8000` port mapping at
all (that mapping is only for reaching it from *outside* Docker, i.e. from
your browser or curl). That's what the `BACKEND_URL=http://backend:8000`
environment variable in `docker-compose.yml` is for.

`depends_on: [backend]` just controls **start order** (Compose starts
`backend` first) — it does not wait for the backend to be ready to serve
requests, only for its container process to have started. That's a fine
simplification for local development; a production setup would add a
proper healthcheck-based wait instead.

### A real gotcha this project hit: unpinned transitive dependencies

`ui/requirements.txt` pins not just `gradio`, but also `huggingface_hub`,
`fastapi`, `starlette`, and `jinja2` — packages `gradio` itself depends on
but that we never call directly. That's not paranoia: while building this,
installing `gradio==4.44.1` alone pulled in the *latest* versions of those
libraries (since `gradio` only declares loose lower-bound constraints on
them), and that combination crashed on startup two different ways (an
`ImportError` from a function `huggingface_hub` had since removed, and a
`TypeError` from a Jinja2/Starlette version mismatch inside Gradio's own
template rendering). Pinning exact versions of a package's key dependencies
— not just the package itself — is a common real-world fix when a library
hasn't been updated to track its dependencies' breaking changes.

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

#### Why "conversation" needs the whole history every time

The model has no built-in memory — each call to `model.generate(...)` is
completely independent of any previous call. What makes something feel like
a back-and-forth *conversation* is that `apply_chat_template` can take a
*list* of `{role, content}` turns (system/user/assistant), not just one
prompt, and format them all into a single block of text with the right
markers — e.g. "here's the system instructions, here's what the user said,
here's what you (the assistant) replied, here's what the user said next" —
before asking the model to predict what comes after that.

So the `/chat` endpoint doesn't give the model memory; it makes the
**caller** responsible for resending the entire conversation so far on every
request, and the model re-reads all of it each time to generate the next
reply. This is exactly how OpenAI's and Anthropic's chat APIs work too. Two
consequences worth knowing:
- The server stays **stateless** — no per-user session data to manage, which
  makes it trivial to scale (any instance can handle any request) and safe
  to restart. This is why it fits `/health`-check-and-replace deployment
  patterns well.
- Cost/latency grows with conversation length, since the model reprocesses
  the whole history every turn. Real systems mitigate this with a KV cache
  across turns, or by truncating/summarizing old messages — out of scope
  for this simple version, but worth knowing about.

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
