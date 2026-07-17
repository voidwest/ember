#!/usr/bin/env python3
"""Capture visual-regression screenshots for key docs pages.

Requires the optional Python Playwright package and installed browser binaries:
    python3 -m pip install playwright
    python3 -m playwright install chromium
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DEFAULT_OUTPUT = Path("/tmp/ember-docs-screenshots")
PAGES = [
    ("/", "home"),
    ("/index.ar.html", "home-ar"),
    ("/ember/", "ember"),
    ("/ember/simd-qwen-gemma/", "simd"),
    ("/research-notes/", "research"),
    ("/research-notes/llama-probing-results.html", "llama-probing"),
    ("/research-notes/llama-probing-results.ar.html", "llama-probing-ar"),
    ("/ember/gemma4-parity-debugging/", "gemma-parity"),
]
THEMES = ("dark", "light")
VIEWPORTS = {
    "desktop": {"width": 1366, "height": 900},
    "mobile": {"width": 390, "height": 844},
}


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


def serve_docs(port: int) -> socketserver.TCPServer:
    handler = lambda *args, **kwargs: QuietHandler(*args, directory=DOCS, **kwargs)
    server = socketserver.TCPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed. Install it to capture screenshots:")
        print("  python3 -m pip install playwright")
        print("  python3 -m playwright install chromium")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    server = serve_docs(args.port)
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for theme in THEMES:
                for viewport_name, viewport in VIEWPORTS.items():
                    page = browser.new_page(viewport=viewport)
                    page.add_init_script(
                        f"localStorage.setItem('voidwest-theme', {theme!r})"
                    )
                    for path, name in PAGES:
                        page.goto(base_url + path, wait_until="networkidle")
                        page.screenshot(
                            path=args.output / f"{name}-{viewport_name}-{theme}.png",
                            full_page=True,
                        )
                    page.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()

    print(f"wrote screenshots to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
