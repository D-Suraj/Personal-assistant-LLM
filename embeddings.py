import time
from functools import lru_cache
import requests


class OpenRouterEmbedder:
    def __init__(self, model_name: str, api_key: str = "", base_url: str = ""):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")

    def encode(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for embedding model.")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload = {
                "model": self.model_name,
                "input": batch,
            }
            for attempt in range(3):
                try:
                    res = requests.post(
                        f"{self.base_url}/embeddings",
                        json=payload,
                        headers=headers,
                        timeout=60,
                    )
                    res.raise_for_status()
                    data = res.json()
                    items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                    vectors.extend([item["embedding"] for item in items])
                    break
                except (requests.exceptions.Timeout, requests.exceptions.RequestException) as exc:
                    if attempt == 2:
                        raise exc
                    time.sleep(2 * (attempt + 1))
        return vectors


@lru_cache(maxsize=4)
def get_embedder(
    model_name: str,
    api_key: str = "",
    base_url: str = "",
) -> OpenRouterEmbedder:
    return OpenRouterEmbedder(model_name, api_key=api_key, base_url=base_url)
