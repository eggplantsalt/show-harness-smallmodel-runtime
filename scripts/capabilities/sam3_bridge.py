#!/usr/bin/env python3
"""Persistent MCP-SSE bridge for the existing OpenETA SAM3 service.

This process is run with the OpenETA SAM3 virtualenv, while Show-Harness remains
independent of the MCP package.  Requests and responses are JSON-lines on stdin/stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client


def _result_payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {"success": False, "error": "sam3_response_missing_json"}


async def _run(url: str) -> None:
    async with sse_client(url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            for raw in sys.stdin:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    request = json.loads(raw)
                    result = await session.call_tool(
                        str(request.get("tool", "segment")),
                        arguments=dict(request.get("arguments") or {}),
                    )
                    payload = _result_payload(result)
                except Exception as exc:  # bridge errors are returned to the caller
                    payload = {
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8773/sse")
    args = parser.parse_args()
    try:
        asyncio.run(_run(args.url))
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"bridge: {exc}"}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
