#!/usr/bin/env python3
"""Regenerate the site text of the Learn pages and the observability page from docs/*.md.

Run from anywhere:  python3 website/scripts/learn_from_docs.py

The site keeps its own rendered copy of this text in website/src/learn/*.html so that no
page links to a markdown file. After editing a source document, run this script, rebuild
the site and review the diff. A link to a document that has a site page is rewritten to
that page; a link to a file with no page keeps its words and loses the link.
"""
import html, pathlib, re, sys

SLUG = {
 "START_HERE.md":"start-here","GLOSSARY.md":"glossary","DECISIONS.md":"decisions","READING_RESULTS.md":"reading-results",
 "WHEN_THINGS_GO_WRONG.md":"when-things-go-wrong","WHAT_SCMLE_CANNOT_DO.md":"what-scmle-cannot-do",
 "ARTIFACTS.md":"artifacts","CLOUD_RELEASE.md":"cloud-release","SERVICE_SLO.md":"service","TUTORIAL.md":"quickstart",
 "TENANT_ISOLATION.md":"tenant-isolation","CLI.md":"cli","SECURITY.md":"threat-model","RELEASING.md":"releasing",
 "LOCAL.md":"local","API_STABILITY.md":"api-stability","BEGINNER_FINDINGS.md":None,
}
BASE = "@@BASE@@"

def gh_slug(t):
    t = re.sub(r'`','',t).lower()
    t = re.sub(r'[^\w\s-]','',t)
    return re.sub(r'\s','-',t.strip())

def link(m):
    text, url = m.group(1), m.group(2)
    if url.startswith('#'): return f'<a href="{url}">{text}</a>'
    mm = re.match(r'(?:\./)?([A-Z_]+\.md)(#.*)?$', url)
    if mm and mm.group(1) in SLUG and SLUG[mm.group(1)]:
        return f'<a href="{BASE}/docs/{SLUG[mm.group(1)]}/{mm.group(2) or ""}">{text}</a>'
    if url.startswith('http'):
        return f'<a href="{url}">{text}</a>'
    return text  # drop links to files that have no page; keep the words

def inline(s):
    codes=[]
    def stash(m):
        codes.append(m.group(1)); return f'\x00{len(codes)-1}\x00'
    s = re.sub(r'`([^`]+)`', stash, s)
    s = html.escape(s, quote=False)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'<em>\1</em>', s)
    s = re.sub(r'\x00(\d+)\x00', lambda m: '<code>'+html.escape(codes[int(m.group(1))], quote=False)+'</code>', s)
    return s

def table(lines):
    rows=[[c.strip() for c in re.split(r'(?<!\\)\|', l.strip().strip('|'))] for l in lines]
    head=rows[0]; body=rows[2:]
    n=len(head)
    mw = '0' if n<=2 else ('560px' if n==3 else '720px')
    label = html.escape(re.sub(r'<[^>]+>','',inline(head[0])) or 'Table')
    out=[f'<div class="model-table" role="region" aria-label="Table: {label}" tabindex="0"><table style="min-width:{mw}"><thead><tr>']
    out += [f'<th scope="col">{inline(h)}</th>' for h in head]
    out.append('</tr></thead><tbody>')
    for r in body:
        r += ['']*(n-len(r))
        out.append('<tr>'+''.join((f'<th scope="row">{inline(c)}</th>' if i==0 and n>2 else f'<td>{inline(c)}</td>') for i,c in enumerate(r[:n]))+'</tr>')
    out.append('</tbody></table></div>')
    return ''.join(out)

def lists(lines):
    # lines: block of list lines; supports one nesting level
    out=[]; stack=[]  # (indent, tag)
    def close_to(ind):
        while stack and stack[-1][0]>=ind:
            out.append('</li></'+stack.pop()[1]+'>')
    first=True
    cur=None
    items=[]
    for l in lines:
        m=re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', l)
        if m:
            items.append([len(m.group(1)), 'ol' if m.group(2)[0].isdigit() else 'ul', m.group(3)])
        else:
            items[-1][2]+=' '+l.strip()
    prev=-1
    for ind,tag,text in items:
        if not stack or ind>stack[-1][0]:
            out.append(f'<{tag}><li>{inline(text)}'); stack.append((ind,tag))
        else:
            while stack and stack[-1][0]>ind:
                out.append('</li></'+stack.pop()[1]+'>')
            out.append(f'</li><li>{inline(text)}')
    while stack: out.append('</li></'+stack.pop()[1]+'>')
    return ''.join(out)

def convert(md, glossary=False):
    L=md.split('\n'); i=0; out=[]; title=None; lead_done=False; seen_h2=False
    used=set()
    while i<len(L):
        l=L[i]
        if l.startswith('```'):
            lang=l[3:].strip().split(' ')[0]; i+=1; buf=[]
            while not L[i].startswith('```'): buf.append(L[i]); i+=1
            i+=1
            out.append('<pre><code'+(f' class="language-{lang}"' if lang else '')+'>'+html.escape('\n'.join(buf),quote=False)+'</code></pre>')
            continue
        m=re.match(r'^(#{1,3}) (.*)$', l)
        if m:
            lvl=len(m.group(1)); t=m.group(2).strip()
            if lvl==1: title=t
            else:
                sid=gh_slug(t); base=sid; n=2
                while sid in used: sid=f'{base}-{n}'; n+=1
                used.add(sid)
                if lvl==2: seen_h2=True
                out.append(f'<h{lvl} id="{sid}">{inline(t)}</h{lvl}>')
            i+=1; continue
        if l.startswith('|'):
            b=[]
            while i<len(L) and L[i].startswith('|'): b.append(L[i]); i+=1
            out.append(table(b)); continue
        if re.match(r'^\s*([-*]|\d+\.)\s+', l):
            b=[]
            while i<len(L) and L[i].strip() and (re.match(r'^\s*([-*]|\d+\.)\s+', L[i]) or L[i].startswith('  ')):
                b.append(L[i]); i+=1
            out.append(lists(b)); continue
        if not l.strip(): i+=1; continue
        b=[]
        while i<len(L) and L[i].strip() and not L[i].startswith(('```','|','#')) and not re.match(r'^\s*([-*]|\d+\.)\s+', L[i]):
            b.append(L[i]); i+=1
        if glossary and b and re.match(r'\*\*(Meaning|Example|Why it matters)\.\*\*', b[0]):
            cur=[]
            for x in b:
                if re.match(r'\*\*(Meaning|Example|Why it matters)\.\*\*', x): cur.append(x)
                else: cur[-1]+=' '+x.strip()
            for x in cur: out.append('<p>'+inline(x)+'</p>')
            continue
        cls=' class="lead"' if (not seen_h2 and not lead_done) else ''
        lead_done=True
        out.append(f'<p{cls}>'+inline(' '.join(x.strip() for x in b))+'</p>')
    return title, '\n'.join(out)+'\n'



ROOT = pathlib.Path(__file__).resolve().parents[2]
JOBS = [  # (source, output name, glossary layout, demote h2 to h3)
    ("START_HERE", "start-here", False, False),
    ("GLOSSARY", "glossary", True, False),
    ("DECISIONS", "decisions", False, False),
    ("READING_RESULTS", "reading-results", False, False),
    ("WHEN_THINGS_GO_WRONG", "when-things-go-wrong", False, False),
    ("WHAT_SCMLE_CANNOT_DO", "what-scmle-cannot-do", False, False),
    ("BEGINNER_FINDINGS", "beginner-findings", False, True),
    ("OPERATIONS", "observability", False, False),
]

if __name__ == "__main__":
    out = ROOT / "website" / "src" / "learn"
    out.mkdir(parents=True, exist_ok=True)
    for src, name, gloss, demote in JOBS:
        title, text = convert((ROOT / "docs" / f"{src}.md").read_text(), gloss)
        if demote:
            text = re.sub(r'<h2 id="([^"]+)">', r'<h3 id="\1">', text).replace("</h2>", "</h3>")
            text = text.replace('<p class="lead">', "<p>")
        if name == "glossary":
            text = text.replace("Use your browser's search to find a word.", "Use the search box above to find a word.")
        (out / f"{name}.html").write_text(text)
        print(f"{name}: {title}")
