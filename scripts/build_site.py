"""Refresh the GitHub Pages site in docs/.

usage: build_site.py   (run after analyze.py)

- docs/guide.html is generated from QUANTIZATION_GUIDE.md, so the two never drift apart.
- docs/index.html and docs/method.html are edited by hand.
- Every figure any page references is copied from figures/ into docs/figures/.
"""

import html
import re
import shutil
import sys

from common import FIGURES, ROOT

DOCS = ROOT / "docs"
REPO = "https://github.com/engineeringthefutureai/bits-of-pi/blob/main/"
PAGES = {"QUANTIZATION_GUIDE.md": "guide.html"}
NAV = [("./", "Results"), ("method.html", "How the experiments work"), ("guide.html", "How the model files were made")]


def nav(current):
    links = "".join(f'<a href="{href}"{" aria-current=\"page\"" if href == current else ""}>{label}</a>'
                    for href, label in NAV)
    return f'  <nav class="site-nav" aria-label="Pages">{links}</nav>\n'


def head(title, description, path):
    """The <head> of index.html with this page's title, description and URL."""
    h = (DOCS / "index.html").read_text().split("<body>")[0]
    h = re.sub(r"<title>[^<]*</title>", f"<title>{title}</title>", h)
    for key in ('name="description"', 'property="og:description"'):
        h = re.sub(rf'<meta {key} content="[^"]*">', f'<meta {key} content="{description}">', h)
    h = re.sub(r'<meta property="og:title" content="[^"]*">', f'<meta property="og:title" content="{title}">', h)
    return h.replace('content="https://bits-of-pi.engineeringthefuture.ai/"',
                     f'content="https://bits-of-pi.engineeringthefuture.ai/{path}"')


def link(target):
    page, _, anchor = target.partition("#")
    if page in PAGES:
        page = PAGES[page]
    elif page.endswith(".md") or page in ("config.yaml",):
        page = REPO + page
    return page + (f"#{anchor}" if anchor else "")


def inline(text):
    """Markdown inline syntax used in the guide: code, bold, italics, links."""
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(f"<code>{html.escape(p[1:-1])}</code>")
            continue
        p = html.escape(p, quote=False)
        p = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f'<a href="{link(m.group(2))}">{m.group(1)}</a>', p)
        p = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", p)
        p = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<em>\1</em>", p)
        out.append(p)
    return "".join(out)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head_, body = cells[0], cells[2:]
    th = "".join(f"<th>{inline(c)}</th>" for c in head_)
    trs = "\n".join("          <tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
    return (f'    <div class="table-wrap">\n      <table>\n        <thead><tr>{th}</tr></thead>\n'
            f"        <tbody>\n{trs}\n        </tbody>\n      </table>\n    </div>\n")


def items(lines):
    """A (possibly nested) list from its lines. Nesting is by indentation."""
    ordered = re.match(r"\s*\d+\.", lines[0]) is not None
    base = len(lines[0]) - len(lines[0].lstrip())
    out, cur = [], None
    for ln in lines:
        indent = len(ln) - len(ln.lstrip())
        m = re.match(r"\s*(?:[-*]|\d+\.)\s+(.*)", ln)
        if m and indent == base:
            cur = [m.group(1), []]
            out.append(cur)
        elif indent > base and re.match(r"\s*(?:[-*]|\d+\.)\s", ln):
            cur[1].append(ln)
        elif cur[1]:
            cur[1][-1] += " " + ln.strip()
        else:
            cur[0] += " " + ln.strip()
    tag = "ol" if ordered else "ul"
    body = "".join(f"<li>{inline(t)}{items(sub) if sub else ''}</li>" for t, sub in out)
    return f'<{tag} class="prose">{body}</{tag}>'


def md_to_body(md):
    """Convert the guide's Markdown into page sections. Returns (title, lede, sections html)."""
    lines = md.splitlines()
    title = lines[0].lstrip("# ").strip()
    i, out, lede, open_section = 1, [], None, False
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif ln.startswith("```"):
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            out.append(f'    <pre class="prompt">{html.escape(chr(10).join(lines[i + 1:j]))}</pre>\n')
            i = j + 1
        elif ln.startswith("## "):
            if open_section:
                out.append("  </section>\n\n")
            text = ln[3:].strip()
            out.append(f'  <section>\n    <h2 id="{slug(text)}">{inline(text)}</h2>\n')
            open_section = True
            i += 1
        elif ln.startswith("### "):
            text = ln[4:].strip()
            out.append(f'    <h3 id="{slug(text)}">{inline(text)}</h3>\n')
            i += 1
        elif ln.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(table(lines[i:j]))
            i = j
        elif ln.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", ln)
            alt, src = m.groups()
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            cap = ""
            if j < len(lines) and lines[j].startswith("*How to read it:*"):
                k = j
                while k < len(lines) and lines[k].strip():
                    k += 1
                text = " ".join(l.strip() for l in lines[j:k]).replace("*How to read it:*", "").strip()
                cap = f"<figcaption><b>How to read it.</b> {inline(text[0].upper() + text[1:])}</figcaption>"
                j = k
            out.append(f'    <figure>\n      <div class="plate"><img src="{src}" alt="{html.escape(alt)}"></div>\n'
                       f"      {cap}\n    </figure>\n")
            i = j
        elif re.match(r"\s*(?:[-*]|\d+\.)\s", ln):
            j = i
            while j < len(lines) and lines[j].strip() and not lines[j].startswith(("#", "|", "```")):
                j += 1
            out.append("    " + items(lines[i:j]) + "\n")
            i = j
        else:
            j = i
            while j < len(lines) and lines[j].strip() and not re.match(r"(#|\||```|!\[|\s*(?:[-*]|\d+\.)\s)", lines[j]):
                j += 1
            para = inline(" ".join(l.strip() for l in lines[i:j]))
            if lede is None and not open_section:
                lede = para
            else:
                out.append(f'    <p class="prose">{para}</p>\n')
            i = j
    if open_section:
        out.append("  </section>\n")
    return title, lede, "".join(out)


def build_guide():
    title, lede, body = md_to_body((ROOT / "QUANTIZATION_GUIDE.md").read_text())
    page = head("How the Model Files Were Made",
                "How every Qwen model file in the study was made: the layers inside Qwen 3.5, what "
                "quantization does to each of them, recipes, and what an imatrix is.", "guide.html")
    page += ("<body>\n\n<main>\n" + nav("guide.html") +
             '  <header style="display:grid; gap:14px;">\n'
             '    <div class="eyebrow">Qwen Digit Recall · background</div>\n'
             f"    <h1>{title}</h1>\n    <p class=\"lede\">{lede}</p>\n  </header>\n\n" + body +
             '\n  <footer>\n    <p>Generated from <a href="' + REPO + 'QUANTIZATION_GUIDE.md">QUANTIZATION_GUIDE.md</a>'
             ' · <a href="https://github.com/engineeringthefutureai/bits-of-pi">github.com/engineeringthefutureai/bits-of-pi</a></p>\n'
             "    <p>Code under the MIT License.</p>\n  </footer>\n</main>\n</body>\n</html>\n")
    (DOCS / "guide.html").write_text(page)


def main():
    build_guide()
    pages = sorted(DOCS.glob("*.html"))
    wanted = sorted({f for p in pages for f in re.findall(r'src="figures/([^"]+)"', p.read_text())})
    missing = [f for f in wanted if not (FIGURES / f).exists()]
    if missing:
        sys.exit(f"figures referenced by docs/*.html but missing from figures/: {missing}")
    out = DOCS / "figures"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for f in wanted:
        shutil.copy2(FIGURES / f, out / f)
    (DOCS / ".nojekyll").touch()
    print(f"docs/: guide.html generated, {len(pages)} pages, {len(wanted)} figures copied")


if __name__ == "__main__":
    main()
