from typing import Any

import requests

from config import APP_NAME, OPENROUTER_BASE_URL


SYSTEM_PROMPT = """You are WorkMind, a friendly personal knowledge-base assistant.
Answer from the provided context whenever it is relevant.
If the context does not contain the answer, say that you could not find it in the indexed files.
Be concise, practical, and cite filenames when helpful."""


def answer_with_openrouter(
    api_key: str,
    model: str,
    question: str,
    context_chunks: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> str:
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured.")

    context = "\n\n".join(
        (
            f"Source: {chunk['metadata'].get('file_path', 'unknown')}\n"
            f"{chunk['document']}"
        )
        for chunk in context_chunks
    )
    if not context:
        context = "No relevant indexed context was found."

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    for item in history or []:
        role = item.get("role", "user")
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": item.get("content", "")})

    messages.append(
        {
            "role": "user",
            "content": (
                "Indexed context:\n"
                f"{context}\n\n"
                f"Question: {question}"
            ),
        }
    )

    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": APP_NAME,
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        },
        timeout=90,
    )

    if not response.ok:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")

    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()
