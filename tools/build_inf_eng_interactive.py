#!/usr/bin/env python3
"""Build a single-file interactive study edition of inf-eng.txt.

The generated HTML contains the complete supplied book text and local study
features. Figures retain their original remote URLs, so text/study tools work
offline while figures require a network connection unless the assets are
separately downloaded by an authorized user.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

from markdown_it import MarkdownIt


def slugify(value: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "section"
    slug = base
    n = 2
    while slug in used:
        slug = f"{base}-{n}"
        n += 1
    used.add(slug)
    return slug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="inf-eng.txt")
    ap.add_argument("output", nargs="?", default="inf-eng-interactive.html")
    args = ap.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    raw = source.read_text(encoding="utf-8")

    md = MarkdownIt("commonmark", {"html": False, "linkify": True, "typographer": True})
    parsed = md.parse(raw)
    heading_rows = []
    used: set[str] = set()
    for idx, token in enumerate(parsed):
        if token.type != "heading_open":
            continue
        level = int(token.tag[1:])
        title = parsed[idx + 1].content.strip()
        heading_rows.append({"level": level, "title": title, "id": slugify(title, used)})

    cursor = iter(heading_rows)
    current = None

    def heading_open(tokens, idx, options, env):
        nonlocal current
        current = next(cursor)
        return f'<{tokens[idx].tag} id="{current["id"]}" tabindex="-1">'

    md.renderer.rules["heading_open"] = heading_open
    rendered = md.render(raw)

    glossary_match = re.search(
        r"^# Appendix A: Inference Glossary\s*(.*?)(?=^# Appendix B:)", raw,
        re.MULTILINE | re.DOTALL,
    )
    glossary = []
    if glossary_match:
        for m in re.finditer(r"^\*\*(.+?):\*\*\s+(.+)$", glossary_match.group(1), re.MULTILINE):
            glossary.append({"term": m.group(1), "definition": m.group(2)})

    toc = []
    for h in heading_rows[1:]:
        if h["level"] > 3:
            continue
        toc.append(
            f'<a class="toc-l{h["level"]}" href="#{h["id"]}" data-title="{html.escape(h["title"].lower())}">'
            f'<span>{html.escape(h["title"])}</span><i aria-hidden="true"></i></a>'
        )

    source_hash = hashlib.sha256(raw.encode()).hexdigest()[:12]
    glossary_json = json.dumps(glossary, ensure_ascii=False).replace("</", "<\\/")
    html_doc = TEMPLATE.replace("__BOOK_HTML__", rendered)
    html_doc = html_doc.replace("__TOC_HTML__", "\n".join(toc))
    html_doc = html_doc.replace("__GLOSSARY_JSON__", glossary_json)
    html_doc = html_doc.replace("__CONCEPT_COUNT__", str(len(glossary)))
    html_doc = html_doc.replace("__SOURCE_HASH__", source_hash)
    output.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size:,} bytes, {len(glossary)} concepts)")


TEMPLATE = r'''<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Inference Engineering — Interactive Study Edition</title>
<style>
:root{--bg:#f6f7fb;--panel:#fff;--ink:#17202a;--muted:#687181;--line:#dce1e8;--brand:#6557e8;--brand2:#16a085;--accent:#fff4cd;--shadow:0 10px 35px rgba(26,33,52,.09);--reading:760px;--font:18px;--radius:16px}
[data-theme=dark]{--bg:#10131a;--panel:#181d27;--ink:#edf0f6;--muted:#a7afbd;--line:#303746;--brand:#9b91ff;--brand2:#51d6b4;--accent:#40371e;--shadow:0 12px 38px rgba(0,0,0,.28)}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:76px}body{margin:0;background:var(--bg);color:var(--ink);font:var(--font)/1.72 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,textarea{font:inherit}button{color:inherit}.skip{position:fixed;left:12px;top:-80px;z-index:100;background:var(--ink);color:var(--bg);padding:10px 14px}.skip:focus{top:12px}.topbar{height:62px;position:fixed;inset:0 0 auto;z-index:30;display:flex;align-items:center;gap:10px;padding:0 18px;background:color-mix(in srgb,var(--panel) 92%,transparent);border-bottom:1px solid var(--line);backdrop-filter:blur(14px)}.brand{font-weight:800;white-space:nowrap}.brand em{font-style:normal;color:var(--brand)}.progress-track{position:absolute;left:0;right:0;bottom:-1px;height:3px}.progress-bar{height:100%;width:0;background:linear-gradient(90deg,var(--brand),var(--brand2))}.nav-tabs{display:flex;gap:4px;margin:auto}.nav-tabs button,.icon-btn,.primary,.secondary{border:1px solid var(--line);background:var(--panel);border-radius:10px;padding:7px 11px;cursor:pointer}.nav-tabs button.active,.primary{background:var(--brand);color:#fff;border-color:var(--brand)}.icon-btn:hover,.secondary:hover,.nav-tabs button:hover{border-color:var(--brand)}.top-actions{display:flex;gap:7px}.layout{display:grid;grid-template-columns:290px minmax(0,1fr);gap:0;padding-top:62px;min-height:100vh}.sidebar{position:fixed;top:62px;bottom:0;width:290px;padding:20px 14px 35px;background:var(--panel);border-right:1px solid var(--line);overflow:auto}.side-title{display:flex;align-items:center;justify-content:space-between;margin:0 7px 10px;font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}.toc-filter,.search-box{width:100%;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:10px;padding:9px 11px;outline:none}.toc-filter:focus,.search-box:focus,textarea:focus{border-color:var(--brand);box-shadow:0 0 0 3px color-mix(in srgb,var(--brand) 18%,transparent)}.toc{margin-top:12px}.toc a{display:flex;align-items:center;gap:7px;color:var(--muted);text-decoration:none;padding:5px 8px;border-radius:7px;font-size:.78rem;line-height:1.3}.toc a:hover,.toc a.active{background:color-mix(in srgb,var(--brand) 10%,transparent);color:var(--brand)}.toc a.active{font-weight:700}.toc a i{margin-left:auto;width:6px;height:6px;border-radius:50%;background:transparent}.toc a.done i{background:var(--brand2)}.toc-l2{padding-left:17px!important}.toc-l3{padding-left:29px!important}.main{grid-column:2;padding:38px 28px 90px;min-width:0}.view{display:none}.view.active{display:block}.book-shell{max-width:var(--reading);margin:auto}.book-meta{background:linear-gradient(135deg,color-mix(in srgb,var(--brand) 12%,var(--panel)),var(--panel));border:1px solid var(--line);border-radius:var(--radius);padding:24px;margin-bottom:38px;box-shadow:var(--shadow)}.book-meta h1{margin:0 0 8px;font-size:1.45rem}.chips{display:flex;gap:8px;flex-wrap:wrap}.chip{font-size:.75rem;padding:4px 9px;border-radius:999px;background:var(--bg);border:1px solid var(--line);color:var(--muted)}.copyright{font-size:.78rem;color:var(--muted);margin-top:14px}.book h1{font-size:2.25rem;line-height:1.12;margin:2.4em 0 .7em;border-top:1px solid var(--line);padding-top:1.15em}.book h1:first-child{border:0;padding-top:0}.book h2{font-size:1.55rem;line-height:1.25;margin:2em 0 .65em}.book h3{font-size:1.18rem;line-height:1.35;margin:1.7em 0 .5em}.book h1,.book h2,.book h3{position:relative}.heading-tools{position:absolute;left:100%;top:0;margin-left:12px;display:flex;gap:4px;opacity:0;transition:.15s}.book h1:hover .heading-tools,.book h2:hover .heading-tools,.book h3:hover .heading-tools,.heading-tools:focus-within{opacity:1}.heading-tools button{border:1px solid var(--line);background:var(--panel);border-radius:7px;cursor:pointer;font-size:.72rem;padding:3px 7px;white-space:nowrap}.book p{margin:0 0 1.15em}.book ul,.book ol{padding-left:1.4em}.book li{margin:.35em 0}.book a{color:var(--brand);text-underline-offset:3px}.book blockquote{margin:1.4em 0;padding:1px 20px;border-left:4px solid var(--brand);color:var(--muted);background:color-mix(in srgb,var(--brand) 5%,transparent)}.book img{display:block;max-width:100%;height:auto;margin:28px auto 8px;border-radius:12px;box-shadow:var(--shadow);background:var(--panel)}.book img.failed{display:none}.image-fallback{padding:18px;border:1px dashed var(--line);border-radius:10px;color:var(--muted);font-size:.85rem}.book code{background:color-mix(in srgb,var(--brand) 10%,transparent);padding:.12em .35em;border-radius:4px}.book pre{overflow:auto;padding:16px;background:#111827;color:#e5e7eb;border-radius:10px}.book hr{border:0;border-top:1px solid var(--line);margin:2em 0}mark.study-hit{background:#ffe37b;color:#151515;border-radius:3px}.panel-head{max-width:1080px;margin:0 auto 26px}.panel-head h1{font-size:2rem;margin:0}.panel-head p{color:var(--muted)}.study-grid{max-width:1080px;margin:auto;display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:22px}.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:22px;box-shadow:var(--shadow)}.concept-controls{display:flex;gap:9px;margin-bottom:16px}.concept-controls input{flex:1}.alpha{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:14px}.alpha button{border:0;background:transparent;color:var(--muted);cursor:pointer;padding:4px}.alpha button:hover{color:var(--brand)}.concept-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;max-height:68vh;overflow:auto;padding-right:5px}.concept{border:1px solid var(--line);border-radius:11px;padding:13px;cursor:pointer;background:var(--bg)}.concept:hover{border-color:var(--brand)}.concept b{display:block;color:var(--brand);margin-bottom:4px}.concept p{font-size:.82rem;line-height:1.45;margin:0;color:var(--muted)}.flashcard{min-height:260px;display:flex;flex-direction:column;justify-content:center;text-align:center;cursor:pointer;perspective:800px}.flash-term{font-weight:800;font-size:1.5rem}.flash-def{display:none;margin-top:18px;color:var(--muted)}.flashcard.revealed .flash-def{display:block}.flash-actions{display:flex;justify-content:center;gap:9px;margin-top:18px}.lab-grid{max-width:1080px;margin:auto;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.lab-card h2{margin-top:0}.lab-card label{display:block;font-size:.78rem;color:var(--muted);margin:10px 0 3px}.lab-card input,.lab-card select{width:100%;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px;padding:8px}.lab-result{margin-top:16px;padding:13px;border-radius:10px;background:color-mix(in srgb,var(--brand) 9%,var(--bg));border:1px solid color-mix(in srgb,var(--brand) 30%,var(--line))}.lab-result strong{display:block;font-size:1.3rem;color:var(--brand)}.lab-explain{font-size:.78rem;color:var(--muted);line-height:1.5}.assumption{font-size:.7rem;color:var(--muted);border-top:1px solid var(--line);padding-top:10px;margin-top:12px}.quiz-wrap{max-width:760px;margin:auto}.question-num{font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.quiz-q{font-size:1.3rem;font-weight:750;margin:12px 0 18px}.options{display:grid;gap:10px}.option{text-align:left;border:1px solid var(--line);border-radius:11px;background:var(--bg);padding:13px;cursor:pointer}.option:hover{border-color:var(--brand)}.option.correct{border-color:#1d9b67;background:color-mix(in srgb,#1d9b67 13%,var(--panel))}.option.wrong{border-color:#d64b4b;background:color-mix(in srgb,#d64b4b 12%,var(--panel))}.quiz-footer{display:flex;align-items:center;justify-content:space-between;margin-top:18px}.modal{border:0;border-radius:16px;background:var(--panel);color:var(--ink);width:min(720px,calc(100% - 28px));max-height:80vh;padding:0;box-shadow:0 25px 80px #0005}.modal::backdrop{background:#080b12aa;backdrop-filter:blur(3px)}.modal-head{display:flex;align-items:center;gap:10px;padding:16px;border-bottom:1px solid var(--line)}.modal-head input{flex:1}.modal-body{padding:12px 16px 20px;overflow:auto;max-height:62vh}.result{display:block;color:var(--ink);text-decoration:none;padding:11px;border-bottom:1px solid var(--line);border-radius:8px}.result:hover{background:var(--bg)}.result b{color:var(--brand)}.result small{display:block;color:var(--muted)}.note-dialog textarea{width:100%;min-height:180px;resize:vertical;border:1px solid var(--line);border-radius:10px;background:var(--bg);color:var(--ink);padding:11px}.note-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:12px}.toast{position:fixed;z-index:80;right:20px;bottom:20px;background:var(--ink);color:var(--bg);border-radius:10px;padding:10px 14px;opacity:0;transform:translateY(10px);pointer-events:none;transition:.2s}.toast.show{opacity:1;transform:none}.mobile-menu{display:none}.empty{text-align:center;color:var(--muted);padding:35px}.footer-note{max-width:var(--reading);margin:50px auto 0;padding-top:18px;border-top:1px solid var(--line);font-size:.75rem;color:var(--muted)}
@media(max-width:900px){.layout{display:block}.sidebar{transform:translateX(-105%);transition:.2s;z-index:28;box-shadow:var(--shadow)}.sidebar.open{transform:none}.main{padding:28px 17px 70px}.mobile-menu{display:inline-block}.brand{font-size:.86rem}.nav-tabs button{padding:6px 8px;font-size:.78rem}.top-actions .font-btn{display:none}.study-grid,.lab-grid{grid-template-columns:1fr}.concept-list{grid-template-columns:1fr}.heading-tools{display:none}}
@media(max-width:560px){.topbar{padding:0 9px}.brand span{display:none}.nav-tabs{margin-left:auto}.top-actions{display:none}.book h1{font-size:1.75rem}.book h2{font-size:1.35rem}.book-shell{width:100%}.book-meta{padding:18px}.concept-list{max-height:none}}
@media print{.topbar,.sidebar,.book-meta,.heading-tools,.footer-note{display:none!important}.layout,.main{display:block;padding:0}.book-shell{max-width:none}.book{font-size:10pt}.book a{color:#000}.book img{max-height:600px}.book h1{break-before:page}}
</style>
</head>
<body>
<a class="skip" href="#reader">Skip to content</a>
<header class="topbar">
<button class="icon-btn mobile-menu" id="menuBtn" aria-label="Open contents">☰</button>
<div class="brand"><em>IE</em> <span>Interactive Study Edition</span></div>
<nav class="nav-tabs" aria-label="Study modes">
<button data-view="reader" class="active">Book</button><button data-view="concepts">Concepts</button><button data-view="labs">Labs</button><button data-view="quiz">Quiz</button>
</nav>
<div class="top-actions"><button class="icon-btn font-btn" id="fontBtn" title="Text size">A±</button><button class="icon-btn" id="searchBtn" title="Search (/)" aria-label="Search">⌕</button><button class="icon-btn" id="themeBtn" title="Theme" aria-label="Toggle theme">◐</button></div>
<div class="progress-track"><div class="progress-bar" id="progress"></div></div>
</header>
<div class="layout">
<aside class="sidebar" id="sidebar" aria-label="Table of contents">
<div class="side-title"><span>Contents</span><span id="readPct">0%</span></div>
<input class="toc-filter" id="tocFilter" type="search" placeholder="Filter sections…" aria-label="Filter table of contents">
<nav class="toc" id="toc">__TOC_HTML__</nav>
</aside>
<main class="main">
<section class="view active" id="reader">
<div class="book-shell">
<div class="book-meta"><h1>Inference Engineering</h1><p>Complete text with navigation, search, notes, progress tracking, a __CONCEPT_COUNT__-term concept explorer, flashcards, and knowledge checks.</p><div class="chips"><span class="chip">Complete text</span><span class="chip">__CONCEPT_COUNT__ concepts</span><span class="chip">Local progress</span><span class="chip">Print friendly</span></div><p class="copyright"><strong>Personal study edition.</strong> The source states: “© 2026 Baseten Labs, Inc. All rights reserved.” Keep this generated copy private unless you have permission to redistribute it. Figures use the publisher’s remote URLs and require internet access.</p></div>
<article class="book" id="book">__BOOK_HTML__</article>
</div>
<div class="footer-note">Built locally from source <code>inf-eng.txt</code> · source fingerprint __SOURCE_HASH__ · study data remains in this browser’s local storage.</div>
</section>
<section class="view" id="concepts">
<div class="panel-head"><h1>Concept Explorer</h1><p>Browse every glossary concept, then use active recall to move terms from “review” to “known.”</p></div>
<div class="study-grid"><div class="card"><div class="concept-controls"><input class="search-box" id="conceptSearch" type="search" placeholder="Search __CONCEPT_COUNT__ concepts…"><button class="secondary" id="randomConcept">Random</button></div><div class="alpha" id="alpha"></div><div class="concept-list" id="conceptList"></div></div><aside class="card"><div class="question-num">Active recall</div><div class="flashcard" id="flashcard" tabindex="0" role="button" aria-label="Reveal flashcard"><div class="flash-term" id="flashTerm"></div><div class="flash-def" id="flashDef"></div><div class="flash-actions" id="flashActions"><button class="secondary" data-rate="review">Review</button><button class="primary" data-rate="known">Known</button></div></div><p id="studyStats" class="copyright"></p></aside></div>
</section>
<section class="view" id="labs">
<div class="panel-head"><h1>Interactive Labs</h1><p>Change one variable at a time, predict the result, then connect the estimate back to the book. These are teaching models—not hardware purchasing or SLA forecasts.</p></div>
<div class="lab-grid">
<div class="card lab-card"><h2>Latency timeline</h2><p class="lab-explain">Separate prefill/time-to-first-token from autoregressive decode.</p><label for="promptTokens">Prompt tokens</label><input id="promptTokens" type="number" min="1" value="512"><label for="outputTokens">Output tokens</label><input id="outputTokens" type="number" min="1" value="128"><label for="prefillRate">Prefill rate (tokens/s)</label><input id="prefillRate" type="number" min="1" value="4000"><label for="decodeRate">Decode rate (tokens/s)</label><input id="decodeRate" type="number" min=".1" step=".1" value="50"><label for="networkMs">Network/queue overhead (ms)</label><input id="networkMs" type="number" min="0" value="80"><div class="lab-result" id="latencyResult"></div><p class="assumption">Simplified serial timeline. Real systems overlap work and show queueing, batching, and percentile variation. See §§1.4 and 2.4.2.</p></div>
<div class="card lab-card"><h2>Roofline explorer</h2><p class="lab-explain">Compare workload arithmetic intensity with a device’s compute-to-bandwidth ratio.</p><label for="peakTops">Peak compute (TOPS)</label><input id="peakTops" type="number" min=".1" value="1000"><label for="bandwidth">Memory bandwidth (GB/s)</label><input id="bandwidth" type="number" min="1" value="3000"><label for="intensity">Arithmetic intensity (ops/byte)</label><input id="intensity" type="number" min=".01" step=".1" value="50"><div class="lab-result" id="roofResult"></div><p class="assumption">Upper-bound roofline estimate: min(peak compute, bandwidth × intensity). Kernel efficiency and memory hierarchy reduce realized performance. See §2.4.1.</p></div>
<div class="card lab-card"><h2>Memory planner</h2><p class="lab-explain">Estimate weight storage plus KV-cache headroom for concurrent requests.</p><label for="paramsB">Parameters (billions)</label><input id="paramsB" type="number" min=".01" step=".1" value="8"><label for="weightBits">Weight precision</label><select id="weightBits"><option value="16">16-bit</option><option value="8" selected>8-bit</option><option value="6">6-bit</option><option value="4">4-bit</option></select><label for="runtimeOverhead">Runtime overhead (%)</label><input id="runtimeOverhead" type="number" min="0" value="20"><label for="kvEach">KV cache per active request (GB)</label><input id="kvEach" type="number" min="0" step=".1" value="1"><label for="concurrency">Concurrent requests</label><input id="concurrency" type="number" min="1" value="8"><div class="lab-result" id="memoryResult"></div><p class="assumption">Illustrative capacity estimate. Quant formats add metadata; KV size depends on architecture, precision, and context length. See §§5.1 and 5.3.</p></div>
</div></section>
<section class="view" id="quiz"><div class="panel-head"><h1>Knowledge Check</h1><p>Questions are generated from the book’s glossary. Each round samples ten concepts.</p></div><div class="card quiz-wrap"><div class="question-num" id="questionNum"></div><div class="quiz-q" id="quizQ"></div><div class="options" id="options"></div><div class="quiz-footer"><span id="score"></span><button class="primary" id="nextQ">Next</button></div></div></section>
</main></div>
<dialog class="modal" id="searchDialog"><div class="modal-head"><input class="search-box" id="globalSearch" type="search" placeholder="Search the complete book…" aria-label="Search book"><button class="icon-btn" id="closeSearch" aria-label="Close">×</button></div><div class="modal-body" id="searchResults"><p class="empty">Type at least two characters.</p></div></dialog>
<dialog class="modal note-dialog" id="noteDialog"><div class="modal-head"><strong id="noteTitle">Section note</strong><button class="icon-btn" id="closeNote" style="margin-left:auto" aria-label="Close">×</button></div><div class="modal-body"><textarea id="noteText" placeholder="Write a private study note…"></textarea><div class="note-actions"><button class="secondary" id="deleteNote">Delete</button><button class="primary" id="saveNote">Save note</button></div></div></dialog>
<div class="toast" id="toast" role="status"></div>
<script id="glossaryData" type="application/json">__GLOSSARY_JSON__</script>
<script>
(()=>{'use strict';
const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
const store={get(k,d){try{return JSON.parse(localStorage.getItem('ie:'+k))??d}catch{return d}},set(k,v){localStorage.setItem('ie:'+k,JSON.stringify(v))}};
const glossary=JSON.parse($('#glossaryData').textContent), views=$$('.view'), tabs=$$('.nav-tabs button');
let currentView='reader', currentFlash=0, noteSection='', quiz=[], qIndex=0, correct=0, answered=false;
function toast(msg){const t=$('#toast');t.textContent=msg;t.classList.add('show');clearTimeout(t._x);t._x=setTimeout(()=>t.classList.remove('show'),1800)}
function showView(id){currentView=id;views.forEach(v=>v.classList.toggle('active',v.id===id));tabs.forEach(t=>t.classList.toggle('active',t.dataset.view===id));$('#sidebar').style.display=id==='reader'?'block':'none';window.scrollTo({top:0,behavior:'smooth'});history.replaceState(null,'','#'+id);if(id==='concepts')renderConcepts();if(id==='quiz'&&!quiz.length)newQuiz()}
tabs.forEach(b=>b.onclick=()=>showView(b.dataset.view));
const savedTheme=store.get('theme',matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');document.documentElement.dataset.theme=savedTheme;
$('#themeBtn').onclick=()=>{const v=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=v;store.set('theme',v)};
const sizes=[16,18,20,22];let fontIndex=store.get('font',1);document.documentElement.style.setProperty('--font',sizes[fontIndex]+'px');
$('#fontBtn').onclick=()=>{fontIndex=(fontIndex+1)%sizes.length;document.documentElement.style.setProperty('--font',sizes[fontIndex]+'px');store.set('font',fontIndex);toast('Text size: '+sizes[fontIndex]+'px')};
$('#menuBtn').onclick=()=>$('#sidebar').classList.toggle('open');
$('#tocFilter').oninput=e=>{$$('#toc a').forEach(a=>a.hidden=!a.dataset.title.includes(e.target.value.toLowerCase()))};
$$('#toc a').forEach(a=>a.onclick=()=>{$('#sidebar').classList.remove('open');showView('reader')});
const headings=$$('#book h1,#book h2,#book h3'), tocLinks=$$('#toc a'), done=new Set(store.get('done',[])), notes=store.get('notes',{});
function addHeadingTools(){headings.forEach(h=>{const tools=document.createElement('span');tools.className='heading-tools';tools.innerHTML='<button class="done-btn" title="Mark section complete">✓</button><button class="note-btn" title="Section note">Note</button><button class="link-btn" title="Copy link">#</button>';h.append(tools);tools.querySelector('.done-btn').onclick=e=>{e.stopPropagation();done.has(h.id)?done.delete(h.id):done.add(h.id);store.set('done',[...done]);syncDone();toast(done.has(h.id)?'Marked complete':'Completion removed')};tools.querySelector('.note-btn').onclick=e=>{e.stopPropagation();openNote(h)};tools.querySelector('.link-btn').onclick=e=>{e.stopPropagation();navigator.clipboard?.writeText(location.href.split('#')[0]+'#'+h.id);toast('Section link copied')};});syncDone()}
function syncDone(){tocLinks.forEach(a=>a.classList.toggle('done',done.has(a.hash.slice(1))))}
function openNote(h){noteSection=h.id;$('#noteTitle').textContent='Note · '+h.childNodes[0].textContent.trim();$('#noteText').value=notes[noteSection]||'';$('#noteDialog').showModal();$('#noteText').focus()}
$('#saveNote').onclick=()=>{const v=$('#noteText').value.trim();if(v)notes[noteSection]=v;else delete notes[noteSection];store.set('notes',notes);$('#noteDialog').close();toast('Note saved')};
$('#deleteNote').onclick=()=>{delete notes[noteSection];store.set('notes',notes);$('#noteDialog').close();toast('Note deleted')};$('#closeNote').onclick=()=>$('#noteDialog').close();addHeadingTools();
function progress(){if(currentView!=='reader')return;const doc=document.documentElement,max=doc.scrollHeight-innerHeight,p=max?Math.min(1,scrollY/max):0;$('#progress').style.width=(p*100)+'%';$('#readPct').textContent=Math.round(p*100)+'%';store.set('progress',scrollY)}addEventListener('scroll',progress,{passive:true});progress();
const observer=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){tocLinks.forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id))}}),{rootMargin:'-15% 0px -75%'});headings.forEach(h=>observer.observe(h));
$$('#book img').forEach(img=>{img.loading='lazy';img.referrerPolicy='no-referrer';img.onerror=()=>{img.classList.add('failed');const p=document.createElement('div');p.className='image-fallback';p.textContent='Figure unavailable offline: '+(img.alt||'open the source URL while online');img.after(p)}});
// Build a compact client-side index from visible semantic blocks.
const blocks=$$('#book h1,#book h2,#book h3,#book p,#book li').map((el,i)=>{if(!el.id)el.id='passage-'+i;return{el,text:el.textContent.replace(/\s+/g,' ').trim(),lower:el.textContent.toLowerCase()}}).filter(x=>x.text.length>2);
function openSearch(){showView('reader');$('#searchDialog').showModal();setTimeout(()=>$('#globalSearch').focus(),30)}$('#searchBtn').onclick=openSearch;$('#closeSearch').onclick=()=>$('#searchDialog').close();
$('#globalSearch').oninput=e=>{const q=e.target.value.trim().toLowerCase(),out=$('#searchResults');if(q.length<2){out.innerHTML='<p class="empty">Type at least two characters.</p>';return}const hits=blocks.filter(b=>b.lower.includes(q)).slice(0,60);out.innerHTML=hits.length?hits.map((b,i)=>{const at=b.lower.indexOf(q),s=Math.max(0,at-65),snippet=b.text.slice(s,at)+ '<b>'+b.text.slice(at,at+q.length)+'</b>'+b.text.slice(at+q.length,at+q.length+110);return '<a class="result" href="#'+b.el.id+'" data-hit="'+i+'">'+snippet+'<small>'+nearestHeading(b.el)+'</small></a>'}).join(''):'<p class="empty">No matches.</p>';$$('.result',out).forEach((a,i)=>a.onclick=()=>{$('#searchDialog').close();setTimeout(()=>highlight(hits[i].el,q),50)})};
function nearestHeading(el){let n=el;while(n&&(n=n.previousElementSibling)){if(/^H[1-3]$/.test(n.tagName))return n.childNodes[0].textContent.trim()}return 'Inference Engineering'}
function highlight(el,q){const w=document.createTreeWalker(el,NodeFilter.SHOW_TEXT);while(w.nextNode()){const n=w.currentNode,i=n.data.toLowerCase().indexOf(q);if(i>=0){const range=document.createRange();range.setStart(n,i);range.setEnd(n,i+q.length);const m=document.createElement('mark');m.className='study-hit';range.surroundContents(m);setTimeout(()=>m.replaceWith(...m.childNodes),2200);break}}el.scrollIntoView({behavior:'smooth',block:'center'})}
addEventListener('keydown',e=>{if(e.key==='/'&&!/INPUT|TEXTAREA/.test(document.activeElement.tagName)){e.preventDefault();openSearch()}if(e.key==='Escape'){$$('#searchDialog[open],#noteDialog[open]').forEach(d=>d.close())}});
// Transparent, deliberately simplified teaching calculators.
function n(id){return Math.max(0,Number($('#'+id).value)||0)}
function updateLabs(){const pre=n('promptTokens')/Math.max(1,n('prefillRate'))*1000,decode=Math.max(0,n('outputTokens')-1)/Math.max(.01,n('decodeRate'))*1000,ttft=n('networkMs')+pre,total=ttft+decode;$('#latencyResult').innerHTML='<strong>'+Math.round(total)+' ms total</strong>TTFT ≈ '+Math.round(ttft)+' ms · decode ≈ '+Math.round(decode)+' ms';const ridge=n('peakTops')*1000/Math.max(1,n('bandwidth')),attain=Math.min(n('peakTops'),n('bandwidth')*n('intensity')/1000),bound=n('intensity')<ridge?'memory-bound':'compute-bound';$('#roofResult').innerHTML='<strong>'+attain.toFixed(1)+' TOPS ceiling</strong>'+bound+' · ridge point '+ridge.toFixed(1)+' ops/byte';const weights=n('paramsB')*n('weightBits')/8,withOverhead=weights*(1+n('runtimeOverhead')/100),kv=n('kvEach')*n('concurrency'),totalMem=withOverhead+kv;$('#memoryResult').innerHTML='<strong>'+totalMem.toFixed(1)+' GB estimated</strong>weights/runtime '+withOverhead.toFixed(1)+' GB · KV '+kv.toFixed(1)+' GB'}
$$('#labs input,#labs select').forEach(x=>x.addEventListener('input',updateLabs));updateLabs();
// Concept explorer and persisted recall ratings.
const ratings=store.get('ratings',{}), alpha='ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');$('#alpha').innerHTML=['All',...alpha].map(x=>'<button data-letter="'+x+'">'+x+'</button>').join('');
function renderConcepts(letter='All'){const q=$('#conceptSearch').value.trim().toLowerCase();const rows=glossary.filter(g=>(letter==='All'||g.term[0].toUpperCase()===letter)&&(g.term+' '+g.definition).toLowerCase().includes(q));$('#conceptList').innerHTML=rows.length?rows.map(g=>'<div class="concept" data-term="'+esc(g.term)+'"><b>'+esc(g.term)+(ratings[g.term]?'<span class="chip" style="float:right">'+ratings[g.term]+'</span>':'')+'</b><p>'+esc(g.definition)+'</p></div>').join(''):'<p class="empty">No concepts found.</p>';$$('.concept').forEach(c=>c.onclick=()=>setFlash(glossary.findIndex(g=>g.term===c.dataset.term)));stats()}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}function setFlash(i){currentFlash=(i+glossary.length)%glossary.length;$('#flashTerm').textContent=glossary[currentFlash].term;$('#flashDef').textContent=glossary[currentFlash].definition;$('#flashcard').classList.remove('revealed')}
function stats(){const known=Object.values(ratings).filter(x=>x==='known').length,review=Object.values(ratings).filter(x=>x==='review').length;$('#studyStats').textContent=known+' known · '+review+' reviewing · '+(glossary.length-known-review)+' unstudied'}
$('#alpha').onclick=e=>{if(e.target.dataset.letter)renderConcepts(e.target.dataset.letter)};$('#conceptSearch').oninput=()=>renderConcepts();$('#randomConcept').onclick=()=>setFlash(Math.floor(Math.random()*glossary.length));$('#flashcard').onclick=e=>{if(!e.target.dataset.rate)$('#flashcard').classList.toggle('revealed')};$('#flashcard').onkeydown=e=>{if(e.key===' '||e.key==='Enter'){$('#flashcard').classList.toggle('revealed');e.preventDefault()}};$('#flashActions').onclick=e=>{e.stopPropagation();if(!e.target.dataset.rate)return;ratings[glossary[currentFlash].term]=e.target.dataset.rate;store.set('ratings',ratings);setFlash((currentFlash+1)%glossary.length);renderConcepts()};setFlash(0);renderConcepts();
// Ten-question multiple-choice rounds.
function shuffled(a){a=[...a];for(let i=a.length-1;i;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
function newQuiz(){quiz=shuffled(glossary).slice(0,10);qIndex=0;correct=0;answered=false;renderQ()}
function renderQ(){if(qIndex>=quiz.length){$('#questionNum').textContent='Round complete';$('#quizQ').textContent='Score: '+correct+' / '+quiz.length;$('#options').innerHTML='<p>'+ (correct>=8?'Excellent recall.':correct>=5?'Good start—review missed concepts.':'Use the Concept Explorer, then try again.')+'</p>';$('#score').textContent='';$('#nextQ').textContent='New round';return}answered=false;const q=quiz[qIndex],distractors=shuffled(glossary.filter(x=>x.term!==q.term)).slice(0,3),opts=shuffled([q,...distractors]);$('#questionNum').textContent='Question '+(qIndex+1)+' of '+quiz.length;$('#quizQ').textContent='Which definition best matches “'+q.term+'”?';$('#options').innerHTML=opts.map(o=>'<button class="option" data-term="'+esc(o.term)+'">'+esc(o.definition)+'</button>').join('');$('#score').textContent='Score: '+correct;$('#nextQ').textContent='Next';$$('.option').forEach(b=>b.onclick=()=>answer(b,q.term))}
function answer(button,term){if(answered)return;answered=true;const ok=button.dataset.term===term;if(ok)correct++;button.classList.add(ok?'correct':'wrong');$$('.option').forEach(b=>{b.disabled=true;if(b.dataset.term===term)b.classList.add('correct')});$('#score').textContent=ok?'Correct · '+correct+' points':'Review this one · '+correct+' points'}
$('#nextQ').onclick=()=>{if(qIndex>=quiz.length){newQuiz();return}if(!answered){toast('Choose an answer first');return}qIndex++;renderQ()};
// Do not restore an old scroll position automatically; retain it for a visible resume prompt.
const old=store.get('progress',0);if(old>500)setTimeout(()=>{const pct=Math.round(old/(document.documentElement.scrollHeight-innerHeight)*100);toast('Previous reading position saved ('+Math.max(1,pct)+'%)')},700);
if(['#concepts','#labs','#quiz'].includes(location.hash))showView(location.hash.slice(1));
})();
</script>
</body></html>'''

if __name__ == "__main__":
    main()
