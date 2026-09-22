"""Tiny interactive chat client for the running server.

Run this on your host machine (not inside Docker) while the container is
running. It keeps the conversation history locally and sends the whole
thing to /chat on every turn — the server itself has no memory between
requests (see README.md).

Usage:
    python chat.py
    python chat.py --url http://localhost:8000
"""

import argparse
import json
import urllib.request

DEFAULT_URL = "http://localhost:8000"


def ask(base_url: str, messages: list[dict]) -> str:
    body = json.dumps({"messages": messages}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["response"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the server")
    args = parser.parse_args()

    print(f"Connected to {args.url}. Type 'exit' to quit.\n")
    messages: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})
        try:
            reply = ask(args.url, messages)
        except Exception as exc:  # noqa: BLE001 - just report it and keep chatting
            print(f"[error] {exc}")
            messages.pop()  # don't keep a turn that failed
            continue

        print(f"Bot: {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
