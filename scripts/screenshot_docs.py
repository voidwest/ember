#!/usr/bin/env python3
"""Capture visual-regression screenshots for key docs pages.

Requires the optional Python Playwright package and installed browser binaries:
    python3 -m pip install playwright
    python3 -m playwright install chromium
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import struct
import sys
import tempfile
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


class DocsServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve_docs(port: int) -> socketserver.TCPServer:
    handler = lambda *args, **kwargs: QuietHandler(*args, directory=DOCS, **kwargs)
    server = DocsServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    args = parser.parse_args()
    if not 0 <= args.port <= 65_535:
        parser.error("--port must be between 0 and 65535 (0 selects a free port)")
    if args.timeout_ms <= 0:
        parser.error("--timeout-ms must be greater than zero")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed. Install it to capture screenshots:")
        print("  python3 -m pip install playwright")
        print("  python3 -m playwright install chromium")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    server = serve_docs(args.port)
    actual_port = int(server.server_address[1])
    base_url = f"http://127.0.0.1:{actual_port}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for theme in THEMES:
                for viewport_name, viewport in VIEWPORTS.items():
                    page = browser.new_page(viewport=viewport)
                    page.set_default_timeout(args.timeout_ms)
                    page.route(
                        "**/*",
                        lambda route: route.continue_()
                        if route.request.url.startswith(base_url)
                        else route.abort(),
                    )
                    page.add_init_script(
                        f"localStorage.setItem('voidwest-theme', {theme!r})"
                    )
                    for path, name in PAGES:
                        response = page.goto(base_url + path, wait_until="domcontentloaded")
                        if response is None or not response.ok:
                            status = response.status if response is not None else "no response"
                            raise RuntimeError(f"failed to load {path}: HTTP {status}")
                        page.evaluate("document.fonts.ready")
                        page.add_style_tag(
                            content="*,*::before,*::after{animation:none!important;transition:none!important}"
                        )
                        payload = page.screenshot(
                            type="png", full_page=True
                        )
                        if not isinstance(payload, bytes):
                            raise RuntimeError(f"browser returned an invalid PNG for {path}")
                        width, height = _png_dimensions(payload)
                        if width != viewport["width"] or height < viewport["height"]:
                            raise RuntimeError(
                                f"browser returned unexpected PNG dimensions for {path}: "
                                f"{width}x{height}"
                            )
                        _atomic_write_bytes(
                            args.output / f"{name}-{viewport_name}-{theme}.png",
                            payload,
                        )
                    page.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()

    print(f"wrote screenshots to {args.output}")
    return 0


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    if (
        len(payload) < 24
        or not payload.startswith(b"\x89PNG\r\n\x1a\n")
        or payload[12:16] != b"IHDR"
    ):
        raise ValueError("invalid PNG header")
    return struct.unpack(">II", payload[16:24])


if __name__ == "__main__":
    sys.exit(main())
