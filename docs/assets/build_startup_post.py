"""Build the bilingual offline post from its two Markdown sources (stdlib only)."""
from pathlib import Path
import html
import re

ROOT = Path(__file__).resolve().parents[1]
STEM = 'where-the-time-goes-before-the-first-token'


def inline(s):
    s = html.escape(s)
    s = re.sub(r'`([^`]+)`', r'<code dir="ltr">\1</code>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', s)


def chart(ar):
    lang = 'ar' if ar else 'en'
    alt = 'Model build: 670 / 749 / 83 ms. Metadata parse: 37.7 / 10.8 ms.' if not ar else 'بناء النموذج: 670 / 749 / 83 مللي ثانية. قراءة البيانات الوصفية: 37.7 / 10.8 مللي ثانية.'
    out = '<figure>'
    for theme in ('dark', 'light'):
        out += '<img class="chart theme-chart-' + theme + '" src="/research-notes/figures/startup-post-' + lang + '-' + theme + '.png" alt="' + alt + '">'
    return out + '</figure>'


def render(text, ar=False):
    # This bounded renderer covers the syntax used by these two source posts.
    lines = text.splitlines(); out=[]; i=0
    while i < len(lines):
        line=lines[i].strip()
        if not line: i+=1; continue
        if line.startswith('![') or line == '<!-- CHART -->':
            out.append(chart(ar)); i+=1; continue
        if line.startswith('```'):
            code=[]; i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]); i+=1
            out.append('<pre dir="ltr"><code>'+html.escape('\n'.join(code))+'</code></pre>'); i+=1; continue
        if line.startswith('#'):
            n=len(line)-len(line.lstrip('#'))
            out.append(f'<h{n}>'+inline(line[n:].strip())+f'</h{n}>'); i+=1; continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', x) for x in cells): rows.append(cells)
                i+=1
            out.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th scope="col">'+inline(x)+'</th>' for x in rows[0])+'</tr></thead><tbody>')
            out.extend('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in row)+'</tr>' for row in rows[1:]); out.append('</tbody></table></div>'); continue
        if line.startswith('- '):
            out.append('<ul>')
            while i<len(lines) and lines[i].strip().startswith('- '):
                part=lines[i].strip()[2:]; i+=1
                while i<len(lines) and lines[i].startswith('  '): part+=' '+lines[i].strip(); i+=1
                out.append('<li>'+inline(part)+'</li>')
            out.append('</ul>'); continue
        parts=[line]; i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','```','|','- ','![')):
            parts.append(lines[i].strip()); i+=1
        out.append('<p>'+inline(' '.join(parts))+'</p>')
    return '\n'.join(out)

# Reuse the site's engineering article shell and its shared theme implementation.
TARGET = ROOT / 'ember/engineering' / STEM
TARGET.mkdir(parents=True, exist_ok=True)
FIGURE_CSS = '''/* Figure-only layout; all colors and typography come from /style.css. */
.startup-figure{padding:20px;direction:inherit}
.startup-grid{display:grid;grid-template-columns:1fr 1fr;gap:32px}
.startup-panel h3{margin:0 0 24px}
.startup-figure .bar-row{margin:22px 0}
.startup-figure .bar-label{display:flex;justify-content:space-between;gap:10px;font-size:13px;margin-bottom:8px}
.startup-figure bdi{white-space:nowrap;font-variant-numeric:tabular-nums;color:var(--heading)}
.startup-figure .track{height:22px;background:var(--surface-2);direction:ltr}
.startup-figure .bar{height:100%;background:var(--muted)}
.startup-figure .bar.after{background:var(--accent)}
.startup-figure .bar.write{background:var(--subtle)}
.startup-figure .axis{display:flex;justify-content:space-between;border-top:1px solid var(--border);padding-top:6px;font-family:var(--mono-font);font-size:11px;color:var(--muted)}
.startup-figure figcaption{margin-top:24px;color:var(--muted);font-size:12px}
@media(max-width:600px){.startup-grid{grid-template-columns:1fr;gap:28px}.startup-figure{padding:16px}}
'''
for value, limit in [(670,900),(749,900),(83,900),(37.7,48),(10.8,48)]:
    FIGURE_CSS += '.startup-figure .value-' + str(value).replace('.', '-') + '{width:' + str(value/limit*100) + '%}\n'
(TARGET / 'figures.css').write_text(FIGURE_CSS)
for ar in (False, True):
    name = 'index.ar.html' if ar else 'index.html'
    template = (ROOT / 'ember/engineering/tool-calling-without-changing-the-inference-core' / name).read_text()
    title = 'فين يروح الوقت قبل أول توكن؟' if ar else 'where the time goes before the first token'
    description = 'تخزين الأوزان المعاد ترتيبها، وقراءة بيانات GGUF الوصفية، وتقليل تخصيص الذاكرة أثناء التوليد.' if ar else 'Packed-weight caching, GGUF metadata parsing, and decode allocations: the measured startup work in Ember.'
    head = template.split('<body>')[0]
    head = re.sub(r'<title>.*?</title>', '<title>'+title+'</title>', head)
    for prop,value in [('og:title',title),('og:description',description),('og:url','https://voidwest.dev/ember/engineering/'+STEM+'/'+('index.ar.html' if ar else ''))]:
        head = re.sub(r'(<meta property="'+prop+r'" content=")[^"]*(")', lambda m:m[1]+html.escape(value,quote=True)+m[2], head)
    # The shared theme script supplies both desktop and mobile controls.
    head = re.sub(r'        <!-- docs:head-scripts start -->.*?<!-- docs:head-scripts end -->\n', '', head, flags=re.S)
    head = head.replace('</head>', '<link rel="stylesheet" href="figures.css">\n    </head>')
    nav = re.search(r'<!-- docs:nav start -->.*?<!-- docs:nav end -->',template,re.S)[0]
    nav = nav.replace('tool-calling-without-changing-the-inference-core',STEM)
    back = re.search(r'<div class="back">.*?</div>',template,re.S)[0]
    footer = re.search(r'<!-- docs:footer start -->.*?<!-- docs:footer end -->',template,re.S)[0]
    source = (ROOT / (STEM + ('.ar.md' if ar else '.md'))).read_text()
    # Phase 2 is local-only (GitHub main returns 404); omit it from public sources.
    source = re.sub(r'- \[Phase 2 optimization report\]\(phase2-optimization-report.md\):.*?(?=\n- |\n\n)', '', source, flags=re.S)
    for document in ('phase3-optimization-report.md', 'v04-execution-contract.md'):
        source = source.replace('](' + document + ')', '](https://github.com/voidwest/ember/blob/main/docs/' + document + ')')
    if not ar:
        source = source[:source.index('The figure is generated')] + 'The HTML figure uses the reported values directly and follows the shared site theme. Rebuild both language pages without running inference:\n\n```sh\npython docs/assets/build_startup_post.py\n```\n'
    # The site uses a separate metadata block and lowercase English headings.
    content = render(source, ar)
    content = re.sub(r'<h1>.*?</h1>\s*<p>.*?</p>', '', content, count=1, flags=re.S)
    if not ar:
        content = re.sub(r'<h2>(.*?)</h2>', lambda m:'<h2>'+m[1].lower()+'</h2>', content)
    content = re.sub(r'href="(?!https?://)([^"/]+\.md)"', r'href="/\1"', content)
    if ar:
        # Isolate visible Latin technical terms and numbers without touching tags.
        chunks = re.split(r'(<[^>]+>)',content)
        depth = 0
        for i, chunk in enumerate(chunks):
            if chunk.startswith('<'):
                if re.match(r'<(?:bdi|code|pre)\b',chunk): depth+=1
                elif re.match(r'</(?:bdi|code|pre)>',chunk): depth-=1
            elif not depth:
                chunks[i] = re.sub(r'[A-Za-z0-9][A-Za-z0-9_.,/+×–%−-]*(?: [A-Za-z0-9][A-Za-z0-9_.,/+×–%−-]*)*',r'<bdi>\g<0></bdi>',chunk)
        content=''.join(chunks)
    meta = '2026-09-12 · Ember · أزمنة بدء التشغيل وحدود القياس' if ar else '2026-09-12 · Ember · startup phases and measurement boundaries'
    page = head + '<body>\n'+nav+'\n'+back+'\n<h1>'+title+'</h1>\n<div class="meta">'+meta+'</div>\n'+content+'\n'+footer+'\n</body>\n</html>\n'
    (TARGET / name).write_text(page)
    # Preserve earlier links while directing readers to the established site lane.
    alias = ROOT / (STEM + ('.ar.html' if ar else '.html'))
    destination = 'ember/engineering/'+STEM+'/'+name
    alias.write_text('<!doctype html>\n<html lang="'+('ar' if ar else 'en')+'"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0; url='+destination+'"><title>'+title+'</title></head><body><a href="'+destination+'">'+title+'</a></body></html>\n')
    print('Built',TARGET/name)

# Apply the existing docs chrome normalizer only to this post's two pages.
import runpy
normalizer = runpy.run_path(str(ROOT.parent / 'scripts/build_docs.py'))
for name in ('index.html', 'index.ar.html'):
    path = TARGET / name
    _, normalized = normalizer['render_file'](path)
    path.write_text(normalized)
