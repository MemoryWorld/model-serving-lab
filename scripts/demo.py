"""Print real SSE chunks; only calls a local gateway unless --url is explicit."""
import argparse
import asyncio
import os

import httpx


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8788")
    parser.add_argument("--model", default="offline-small")
    args = parser.parse_args()
    key = os.getenv("API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(base_url=args.url, trust_env=False, timeout=15) as client:
        async with client.stream("POST", "/v1/chat/completions", headers=headers, json={
            "model": args.model, "stream": True, "max_tokens": 32,
            "messages": [{"role": "user", "content": "Explain admission control for a fictional bookstore assistant."}],
        }) as response:
            response.raise_for_status()
            print("request_id:", response.headers.get("x-request-id"))
            async for line in response.aiter_lines():
                if line:
                    print(line, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
