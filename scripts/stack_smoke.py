"""Canary calls through real HTTP; --verify-only reuses existing persisted fixtures."""

import argparse
import json
import os

import httpx


def verify(gateway, retrieval, key="", *, model="offline-small", write=True):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    evidence = {}
    with httpx.Client(headers=headers, timeout=70, trust_env=False) as client:
        response = client.get(f"{gateway}/readyz")
        response.raise_for_status()
        assert response.json()["ready"], "Model backend is not ready"
        request = {"model": model, "messages": [{"role": "user", "content": "Reply briefly: service ready."}],
                   "max_tokens": 32}
        completion = client.post(f"{gateway}/v1/chat/completions", json=request)
        completion.raise_for_status()
        assert completion.json()["choices"][0]["message"]["content"].strip()
        streamed = client.post(f"{gateway}/v1/chat/completions", json={**request, "stream": True})
        streamed.raise_for_status()
        assert "[DONE]" in streamed.text and "event: error" not in streamed.text
        evidence["inference"] = {"model": model, "non_streaming": True, "sse_done": True}
        response = client.get(f"{retrieval}/readyz")
        response.raise_for_status()
        evidence["retrieval"] = response.json()
        path = f"{retrieval}/v1/namespaces/deployment-canary"
        if write:
            for doc_id, content in (("fire", "Fictional manual: fire exit door must remain clear."),
                                    ("solar", "Fictional manual: solar panels require inspection.")):
                response = client.put(f"{path}/documents", json={"id": doc_id, "title": "Fictional example",
                                      "content": content, "source": "synthetic://deployment-canary"})
                response.raise_for_status()
        result = client.post(f"{path}/search", json={"text": "fire exit door", "top_k": 1})
        result.raise_for_status()
        assert result.json()["matches"][0]["id"] == "fire", "Persisted retrieval canary failed"
        evidence["persisted_document_found"] = True
        invalid = client.post(f"{path}/search", json={"embedding": [1.0], "top_k": 1})
        assert invalid.status_code == 422, "Dimension guard failed"
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="http://127.0.0.1:18877/gateway")
    parser.add_argument("--retrieval", default="http://127.0.0.1:18876")
    parser.add_argument("--model", default="offline-small")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.gateway, args.retrieval, os.getenv("API_KEY", ""),
                            model=args.model, write=not args.verify_only), indent=2))
