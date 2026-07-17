#!/usr/bin/env python3
"""Regenerate shared docs HTML fragments in-place.

The docs site is intentionally static HTML. This script keeps it that way while
making the repeated navigation and syntax-highlighting blocks generated.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
STYLESHEET_VERSION = "20260718"

MANAGED_HEAD_RE = re.compile(
    r"\n[ \t]*<!-- docs:head-scripts start -->.*?[ \t]*<!-- docs:head-scripts end -->\n?",
    re.DOTALL,
)
MANAGED_THEME_RE = re.compile(
    r"\n[ \t]*<!-- docs:theme-script start -->.*?[ \t]*<!-- docs:theme-script end -->\n?",
    re.DOTALL,
)
MANAGED_OG_RE = re.compile(
    r"\n[ \t]*<!-- docs:og-image start -->.*?[ \t]*<!-- docs:og-image end -->\n?",
    re.DOTALL,
)
MANAGED_NAV_RE = re.compile(
    r"\n[ \t]*<!-- docs:nav start -->.*?[ \t]*<!-- docs:nav end -->\n?",
    re.DOTALL,
)
MANAGED_FOOTER_RE = re.compile(
    r"\n[ \t]*<!-- docs:footer start -->.*?[ \t]*<!-- docs:footer end -->\n?",
    re.DOTALL,
)
LEGACY_HLJS_RE = re.compile(
    r"\n\s*<script\b[^>]*highlight\.min\.js[^<]*</script>\s*"
    r"\n\s*<script\b[^>]*languages/rust\.min\.js[^<]*</script>\s*"
    r"\n\s*<script\b[^>]*languages/python\.min\.js[^<]*</script>\s*"
    r"\n\s*<script defer>\s*document\.addEventListener\([\s\S]*?"
    r"hljs\.highlightElement\(el\);[\s\S]*?</script>\s*",
    re.MULTILINE,
)
NAV_RE = re.compile(r"\n\s*<nav class=\"site-nav\"[\s\S]*?</nav>\s*", re.MULTILINE)
FOOTER_RE = re.compile(r"\n\s*<footer>[\s\S]*?</footer>\s*", re.MULTILINE)
HEAD_CLOSE_RE = re.compile(r"\n\s*</head>")
STYLESHEET_RE = re.compile(r'(\n[ \t]*<link rel="stylesheet" href="/style\.css(?:\?v=\d+)?" />)')
BODY_OPEN_RE = re.compile(r"(\n[ \t]*<body>)\s*\n")
BODY_CLOSE_RE = re.compile(r"\n\s*</body>")


THEME_SCRIPT = """\
        <!-- docs:theme-script start -->
        <script src="/theme.js"></script>
        <!-- docs:theme-script end -->
"""


HEAD_SCRIPTS = """\
        <!-- docs:head-scripts start -->
        <script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        <script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/rust.min.js"></script>
        <script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
        <script defer>
            document.addEventListener('DOMContentLoaded', function() {
                document.querySelectorAll('pre code').forEach(function(el) {
                    hljs.highlightElement(el);
                });
            });
        </script>
        <!-- docs:head-scripts end -->
"""


def is_arabic(text: str) -> bool:
    return 'lang="ar"' in text or 'dir="rtl"' in text


def section_for(path: Path) -> str:
    rel = path.relative_to(DOCS).as_posix()
    if rel == "index.html" or rel == "index.ar.html":
        return "home"
    if rel.startswith("ember/"):
        return "ember"
    if rel.startswith("research-notes/"):
        return "research"
    if rel.startswith("tools/"):
        return "tools"
    if rel.startswith("terms/"):
        return "terms"
    return "home"


def alternate_href(path: Path, ar: bool) -> str:
    rel = path.relative_to(DOCS).as_posix()
    if ar:
        if rel == "index.ar.html":
            return "/"
        if rel.endswith("/index.ar.html"):
            return "/" + rel.removesuffix("index.ar.html")
        return "/" + rel.replace(".ar.html", ".html")

    if rel == "index.html":
        return "/index.ar.html"
    if rel.endswith("/index.html"):
        alternate = rel.removesuffix("index.html") + "index.ar.html"
    else:
        alternate = rel.replace(".html", ".ar.html")
    if not (DOCS / alternate).exists():
        return "/index.ar.html"
    return "/" + alternate


def localized_href(section: str, ar: bool) -> str:
    paths = {
        "home": ("/", "/index.ar.html"),
        "ember": ("/ember/", "/ember/index.ar.html"),
        "research": ("/research-notes/", "/research-notes/index.ar.html"),
        "tools": ("/tools/", "/tools/index.ar.html"),
        "terms": ("/terms/", "/terms/index.ar.html"),
    }
    return paths[section][1 if ar else 0]


def og_slug_for(path: Path) -> str:
    rel = path.relative_to(DOCS).as_posix()
    if rel in {"index.html", "index.ar.html"}:
        return "voidwest"
    if rel.startswith("ember/simd-qwen-gemma/"):
        return "simd-qwen-gemma"
    if rel.startswith("ember/"):
        return "ember"
    if rel.startswith("research-notes/llama-probing-results"):
        return "llama-probing-results"
    if rel.startswith("research-notes/"):
        return "research-notes"
    return "voidwest"


def og_image_html(path: Path) -> str:
    url = f"https://voidwest.dev/og/{og_slug_for(path)}.png"
    return f"""\
        <!-- docs:og-image start -->
        <meta property="og:image" content="{url}" />
        <meta property="og:image:width" content="1200" />
        <meta property="og:image:height" content="630" />
        <meta name="twitter:image" content="{url}" />
        <!-- docs:og-image end -->
"""


def nav_html(path: Path, text: str) -> str:
    ar = is_arabic(text)
    current = section_for(path)
    lang_text = "en" if ar else "عربي"
    links = [
        ("ember", "ember", localized_href("ember", ar)),
        ("research", "research notes", localized_href("research", ar)),
        ("tools", "tools", localized_href("tools", ar)),
        ("terms", "terms", localized_href("terms", ar)),
    ]

    def current_attr(section: str) -> str:
        return ' aria-current="page"' if section == current else ""

    link_lines = "\n".join(
        f'                <a href="{href}"{current_attr(section)}>{label}</a>'
        for section, label, href in links
    )
    return f"""\
        <!-- docs:nav start -->
        <nav class="site-nav" aria-label="Primary">
            <a class="brand" href="{localized_href('home', ar)}"{current_attr('home')}>voidwest</a>
            <div class="nav-links">
{link_lines}
            </div>
            <div class="nav-actions">
                <a class="nav-lang" href="{alternate_href(path, ar)}">{lang_text}</a>
                <button class="theme-toggle" type="button" aria-label="Switch to light theme" aria-pressed="false">light</button>
            </div>
        </nav>
        <!-- docs:nav end -->
"""


def footer_html(text: str) -> str:
    ar = is_arabic(text)
    home = localized_href("home", ar)
    ember = localized_href("ember", ar)
    research = localized_href("research", ar)
    tools = localized_href("tools", ar)
    terms = localized_href("terms", ar)
    return f"""\
        <!-- docs:footer start -->
        <footer>
            <a href="{home}">voidwest</a>
            &nbsp;·&nbsp;
            <a href="https://github.com/voidwest">github</a>
            &nbsp;·&nbsp;
            <a href="https://www.linkedin.com/in/mthobaiti/" rel="me">linkedin</a>
            &nbsp;·&nbsp;
            <a href="mailto:mthobaiti@outlook.com">email</a>
            &nbsp;·&nbsp;
            <a href="{ember}">ember</a>
            &nbsp;·&nbsp;
            <a href="{research}">research notes</a>
            &nbsp;·&nbsp;
            <a href="{tools}">tools</a>
            &nbsp;·&nbsp;
            <a href="{terms}">terms</a>
        </footer>
        <!-- docs:footer end -->
"""


def update_head_scripts(text: str) -> str:
    text = MANAGED_HEAD_RE.sub("\n", text)
    text = LEGACY_HLJS_RE.sub("\n", text)
    if "<pre" not in text:
        return text
    return HEAD_CLOSE_RE.sub("\n" + HEAD_SCRIPTS + "    </head>", text, count=1)


def update_theme_script(text: str) -> str:
    text = MANAGED_THEME_RE.sub("\n", text)
    return HEAD_CLOSE_RE.sub("\n" + THEME_SCRIPT + "    </head>", text, count=1)


def update_og_image(path: Path, text: str) -> str:
    text = MANAGED_OG_RE.sub("\n", text)
    return STYLESHEET_RE.sub(r"\1\n" + og_image_html(path).rstrip(), text, count=1)


def update_stylesheet(text: str) -> str:
    href = f'\n        <link rel="stylesheet" href="/style.css?v={STYLESHEET_VERSION}" />'
    return STYLESHEET_RE.sub(href, text, count=1)


def update_nav(path: Path, text: str) -> str:
    text = MANAGED_NAV_RE.sub("\n", text)
    text = NAV_RE.sub("\n", text, count=1)
    return BODY_OPEN_RE.sub(r"\1\n" + nav_html(path, text) + "\n", text, count=1)


def update_footer(text: str) -> str:
    text = MANAGED_FOOTER_RE.sub("\n", text)
    text = FOOTER_RE.sub("\n", text, count=1)
    return BODY_CLOSE_RE.sub("\n" + footer_html(text) + "\n    </body>", text, count=1)


def update_file(path: Path) -> bool:
    old = path.read_text()
    new = update_head_scripts(old)
    new = update_theme_script(new)
    new = update_stylesheet(new)
    new = update_og_image(path, new)
    new = update_nav(path, new)
    new = update_footer(new)
    if new == old:
        return False
    if "--check" not in sys.argv:
        path.write_text(new)
    return True


def main() -> int:
    changed = [path for path in sorted(DOCS.rglob("*.html")) if update_file(path)]
    for path in changed:
        print(path.relative_to(ROOT))
    verb = "would update" if "--check" in sys.argv else "updated"
    print(f"{verb} {len(changed)} file(s)")
    return 1 if changed and "--check" in sys.argv else 0


if __name__ == "__main__":
    sys.exit(main())
