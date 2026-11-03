"""Refresh the GitHub Pages site in docs/.

usage: build_site.py   (run after analyze.py)

- docs/guide.html is generated from QUANTIZATION_GUIDE.md, so the two never drift apart.
- docs/index.html and docs/method.html are edited by hand.
- Every figure any page references is copied from figures/ into docs/figures/.
- Search-engine files are generated: canonical links, structured data and image sizes in every page,
  plus robots.txt, sitemap.xml, llms.txt and 404.html.
"""

import csv
import html
import json
import math
import re
import shutil
import struct
import sys
import zlib

from common import FIGURES, ROOT, SCORES, TF, load_config, read_jsonl

DOCS = ROOT / "docs"
SITE = "https://bits-of-pi.engineeringthefuture.ai/"
GITHUB = "https://github.com/engineeringthefutureai/bits-of-pi"
REPO = GITHUB + "/blob/main/"
NAME = "Qwen Digit Recall"
SHARE_IMAGE = "og.png"  # link-preview card, kept in docs/
# Social previews show about 125 characters, so og:description is shorter than the description Google shows.
OG_DESCRIPTIONS = {
    "index.html": "How quantization affects π recall: 64 GGUF versions of Qwen 3.5 4B, 9B and Qwen 3.8 27B, from BF16 down to 2 bits.",
    "method.html": "The seven tests behind Qwen Digit Recall, with real prompts and answers, and teacher forcing explained step by step.",
    "guide.html": "How every model file in the study was made: Qwen 3.5's layers, quantization types, recipes and the imatrix.",
    "data.html": "Every number behind Qwen Digit Recall: 64 GGUF versions, every scored answer, and per-digit certainty for π.",
}
KEYWORDS = ["quantization", "GGUF", "llama.cpp", "Qwen 3.5", "Qwen 3.8", "digits of pi", "memorization",
            "KL divergence", "teacher forcing", "imatrix", "K-quants", "bits per weight"]
PAGES = {"QUANTIZATION_GUIDE.md": "guide"}
# Pages are linked without ".html": GitHub Pages serves method.html at /method.
NAV = [("./", "Results"), ("method", "Method"), ("guide", "Guide"), ("data", "Raw data")]


def nav(current, root=""):
    """The sticky top bar: brand, page links and GitHub. root="/" gives absolute links (for 404.html)."""
    def url(href):
        return (root or "./") if href == "./" else root + href
    links = "".join(f'<a href="{url(href)}"{" aria-current=\"page\"" if href == current else ""}>{label}</a>'
                    for href, label in NAV)
    return ('<header class="topbar"><div class="topbar-in">'
            f'<a class="brand" href="{url("./")}"><span class="pi">π</span><span>Bits of Pi</span>'
            f'<span class="sep">/</span><span class="sub">{NAME}</span></a>'
            f'<nav class="site-nav" aria-label="Pages">{links}<a class="ext" href="{GITHUB}">GitHub</a></nav>'
            "</div></header>\n")


def set_topbar(page):
    """Put the current top bar into a hand-edited page."""
    current = "./" if page.name == "index.html" else page.stem
    text = re.sub(r'<header class="topbar">.*?</header>\n|  <nav class="site-nav"[^\n]*</nav>\n', "",
                  page.read_text())
    page.write_text(text.replace("<body>\n", "<body>\n" + nav(current), 1))


def head(title, description, path):
    """The <head> of index.html with this page's title, description and URL."""
    h = (DOCS / "index.html").read_text().split("<body>")[0]
    h = re.sub(r"<!-- seo -->.*?<!-- /seo -->\n", "", h, flags=re.S)
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
    page = head(f"How the GGUF Model Files Were Made · {NAME}",
                "How every Qwen model file in the study was made: the layers inside Qwen 3.5, what "
                "quantization does to each of them, recipes, and what an imatrix is.", "guide")
    page += ("<body>\n" + nav("guide") + "\n<main>\n" +
             '  <header style="display:grid; gap:14px;">\n'
             '    <div class="eyebrow">Qwen Digit Recall · background</div>\n'
             f"    <h1>{title}</h1>\n    <p class=\"lede\">{lede}</p>\n  </header>\n\n" + body +
             '\n  <footer>\n    <p>Generated from <a href="' + REPO + 'QUANTIZATION_GUIDE.md">QUANTIZATION_GUIDE.md</a>'
             ' · <a href="https://github.com/engineeringthefutureai/bits-of-pi">github.com/engineeringthefutureai/bits-of-pi</a></p>\n'
             "    <p>Code under the MIT License.</p>\n  </footer>\n</main>\n</body>\n</html>\n")
    (DOCS / "guide.html").write_text(page)


def num(x, digits=4):
    """Round for the browser: floats to `digits` significant figures, NaN/None to None."""
    if x is None or x == "" or x == "None":
        return None
    x = float(x)
    if math.isnan(x):
        return None
    if x == int(x) and abs(x) < 1e15:
        return int(x)
    return float(f"{x:.{digits}g}")


def error_list(errors, limit=40):
    """'S@10 I@94 …' from a scored record's alignment errors (op and position in the reference)."""
    if not errors:
        return ""
    s = " ".join(f"{e['op']}@{e['ref_pos']}" for e in errors[:limit])
    return s + (f" … (+{len(errors) - limit})" if len(errors) > limit else "")


def export_data():
    """docs/downloads/: the headline table of every model, every scored test answer and per-digit
    teacher forcing on π, as compact JSON for data.html, plus the headline CSVs for download.
    Model outputs are not included: they live only in the raw run logs, which are not published."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        out = DOCS / "downloads"
        if out.exists() and any(out.iterdir()):
            return sorted(f.name for f in out.iterdir())
        raise
    out = DOCS / "downloads"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    cfg = load_config()
    versions = {"cols": None, "rows": []}
    for model in cfg["models"]:
        p = SCORES / f"{model}_headline.csv"
        if not p.exists():
            continue
        shutil.copy2(p, out / p.name)
        with open(p) as f:
            rows = list(csv.DictReader(f))
        cols = ["model"] + list(rows[0])
        versions["cols"] = versions["cols"] or cols
        for r in rows:
            versions["rows"].append([model] + [r[c] if c in ("variant", "group") else num(r[c])
                                               for c in cols[1:]])

        recs = read_jsonl(SCORES / f"{model}.jsonl")
        rcols = ["variant", "exp", "input_id", "meta", "prefix", "errors_per_100", "edit_distance",
                 "subs", "ins", "dels", "outcome", "parse_errors", "hyp_len", "tokens_out", "finish_reason",
                 "errors"]
        runs = [[r["variant"], r["exp"], r["input_id"],
                 " ".join(f"{k}={v}" for k, v in r.get("meta", {}).items()),
                 r.get("prefix"), num(r.get("errors_per_100")), r.get("edit_distance"), r.get("subs"),
                 r.get("ins"), r.get("dels"), r.get("outcome"), r.get("parse_errors"), r.get("hyp_len"),
                 r.get("tokens_out"), r.get("finish_reason"), error_list(r.get("errors"))] for r in recs]
        (out / f"runs_{model}.json").write_text(json.dumps({"cols": rcols, "rows": runs}, separators=(",", ":")))

        # per-digit teacher forcing on π, in the order of the headline table (most bits first)
        digits, tf = None, []
        for r in rows:
            q = TF / model / r["variant"] / "pi.parquet"
            if not q.exists():
                continue
            d = pq.read_table(q).to_pandas()
            digits = digits or "".join(d["digit"])
            tf.append({"variant": r["variant"], "group": r["group"], "bpw": num(r["bpw"]),
                       "p": [num(math.exp(x), 3) for x in d["logprob"]],
                       "rank": [int(x) for x in d["rank"]],
                       "margin": [num(x, 3) for x in d["margin"]],
                       "kl": [num(x, 3) for x in d["kl_vs_ref"]]})
        (out / f"pi_{model}.json").write_text(json.dumps({"digits": digits, "versions": tf}, separators=(",", ":")))
    (out / "versions.json").write_text(json.dumps(versions, separators=(",", ":")))
    return sorted(f.name for f in out.iterdir())


def png_size(path):
    with open(path, "rb") as f:
        return struct.unpack(">II", f.read(24)[16:24])


def page_url(name):
    return SITE if name == "index.html" else SITE + name.removesuffix(".html")


def structured_data(name, title, description, author, files):
    """schema.org JSON-LD: the site, plus an article or, for the raw data page, a dataset."""
    site = {"@type": "WebSite", "@id": SITE + "#site", "name": NAME, "alternateName": "Bits of Pi", "url": SITE}
    person = {"@type": "Person", "name": author, "url": GITHUB}
    common = {"name": title, "description": description, "url": page_url(name), "inLanguage": "en",
              "author": person, "isPartOf": {"@id": SITE + "#site"}, "keywords": ", ".join(KEYWORDS)}
    if name == "data.html":
        downloads = [{"@type": "DataDownload", "contentUrl": SITE + "downloads/" + f,
                      "encodingFormat": "text/csv" if f.endswith(".csv") else "application/json"}
                     for f in sorted(files)]
        item = {"@type": "Dataset", **common, "creator": person, "isAccessibleForFree": True,
                "license": REPO + "LICENSE", "sameAs": GITHUB, "distribution": downloads,
                "measurementTechnique": "Greedy generation and teacher-forced logit dumps with llama.cpp",
                "variableMeasured": ["digits of pi, e and square root of 2 recalled", "digits copied correctly",
                                     "per-digit logit margin", "KL divergence from BF16", "VRAM", "tokens per second"]}
        del item["author"]
    else:
        item = {"@type": "Article" if name == "index.html" else "TechArticle", **common, "headline": title,
                "image": SITE + SHARE_IMAGE}
    return json.dumps({"@context": "https://schema.org", "@graph": [site, item]}, ensure_ascii=False, indent=1)


def seo(page, files):
    """Rewrite one page's search-engine block in <head> and give every image its size."""
    text = page.read_text()
    text = re.sub(r"<!-- seo -->.*?<!-- /seo -->\n", "", text, flags=re.S)
    title = html.unescape(re.search(r"<title>([^<]*)</title>", text)[1])
    description = html.unescape(re.search(r'<meta name="description" content="([^"]*)">', text)[1])
    author = re.search(r'<meta name="author" content="([^"]*)">', text)[1]
    w, h = png_size(DOCS / SHARE_IMAGE)
    img_alt = ("Qwen Digit Recall: digits of π recited correctly against bits per weight for Qwen 3.5 4B, "
               "Qwen 3.5 9B and Qwen 3.8 27B. All three lose digits at Q4_K_M, about 5 bits.")
    og_desc = OG_DESCRIPTIONS.get(page.name, description)
    block = (f'<!-- seo -->\n<link rel="canonical" href="{page_url(page.name)}">\n'
             f'<meta property="og:site_name" content="{NAME}">\n'
             '<meta property="og:locale" content="en_US">\n'
             f'<meta property="og:image:width" content="{w}">\n<meta property="og:image:height" content="{h}">\n'
             f'<meta property="og:image:alt" content="{img_alt}">\n'
             f'<meta name="twitter:title" content="{html.escape(title, quote=True)}">\n'
             f'<meta name="twitter:description" content="{html.escape(og_desc, quote=True)}">\n'
             f'<meta name="twitter:image" content="{SITE}{SHARE_IMAGE}">\n'
             f'<meta name="twitter:image:alt" content="{img_alt}">\n'
             '<meta name="theme-color" content="#2a78d6">\n'
             '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
             '<script type="application/ld+json">\n'
             + structured_data(page.name, title, description, author, files) + "\n</script>\n<!-- /seo -->\n")
    text = text.replace("</head>", block + "</head>", 1)
    icon_tags = (
        '<link rel="icon" type="image/svg+xml" sizes="any" href="/favicon.svg">\n'
        '<link rel="icon" href="/favicon.ico" sizes="48x48">\n'
        '<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">'
    )
    text = re.sub(r'(<link rel="(?:icon|apple-touch-icon)"[^>]*>\n?)+', icon_tags + "\n", text, count=1)
    if page.name in OG_DESCRIPTIONS:
        text = re.sub(r'<meta property="og:description" content="[^"]*">',
                      f'<meta property="og:description" content="{OG_DESCRIPTIONS[page.name]}">', text, count=1)

    def sized(m):
        tag = re.sub(r' (width|height|loading|decoding)="[^"]*"', "", m[0])
        w, h = png_size(FIGURES / m[1])
        return tag[:-1] + f' width="{w}" height="{h}" loading="lazy" decoding="async">'
    page.write_text(re.sub(r'<img src="figures/([^"]+)"[^>]*>', sized, text))


def not_found():
    page = head(f"Page not found · {NAME}", "This page does not exist.", "404")
    page = page.replace('href="style.css"', 'href="/style.css"').replace(
        '<meta name="viewport"', '<meta name="robots" content="noindex">\n<meta name="viewport"')
    page += ("<body>\n" + nav(None, root="/") + "\n<main>\n" +
             '  <header style="display:grid; gap:14px;">\n'
             f'    <div class="eyebrow">{NAME}</div>\n    <h1>Page not found</h1>\n'
             '    <p class="lede">There is nothing at this address. The results, the method, the guide to the model '
             'files and the raw data are linked above.</p>\n  </header>\n</main>\n</body>\n</html>\n')
    (DOCS / "404.html").write_text(page)


def _cubic_bezier(p0, p1, p2, p3, steps=24):
    pts = []
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def _pi_polygon():
    """Closed polygon approximation of the Georgia π glyph in 64x64 viewBox coordinates."""
    poly = [(47.62, 23.09), (47.62, 26.86), (41.36, 26.86)]
    poly += _cubic_bezier((41.36, 26.86), (41.22, 28.60), (41.05, 31.15), (40.85, 34.50))
    poly += _cubic_bezier((40.85, 34.50), (40.65, 37.84), (40.55, 39.89), (40.55, 40.66))
    poly += _cubic_bezier((40.55, 40.66), (40.55, 41.80), (40.82, 42.59), (41.35, 43.02))
    poly += _cubic_bezier((41.35, 43.02), (41.87, 43.46), (42.50, 43.67), (43.24, 43.67))
    poly += _cubic_bezier((43.24, 43.67), (44.03, 43.67), (44.71, 43.46), (45.28, 43.05))
    poly += _cubic_bezier((45.28, 43.05), (45.85, 42.63), (46.50, 41.94), (47.25, 40.96))
    poly.append((48.10, 41.80))
    poly += _cubic_bezier((48.10, 41.80), (47.22, 43.39), (46.18, 44.61), (44.99, 45.46))
    poly += _cubic_bezier((44.99, 45.46), (43.80, 46.31), (42.36, 46.73), (40.68, 46.73))
    poly += _cubic_bezier((40.68, 46.73), (38.92, 46.73), (37.65, 46.23), (36.88, 45.23))
    poly += _cubic_bezier((36.88, 45.23), (36.11, 44.23), (35.72, 42.63), (35.72, 40.42))
    poly += _cubic_bezier((35.72, 40.42), (35.72, 39.46), (35.85, 37.39), (36.11, 34.21))
    poly += _cubic_bezier((36.11, 34.21), (36.37, 31.02), (36.57, 28.57), (36.71, 26.86))
    poly.append((28.88, 26.86))
    poly += _cubic_bezier((28.88, 26.86), (28.54, 33.43), (28.11, 38.29), (27.58, 41.45))
    poly += _cubic_bezier((27.58, 41.45), (27.05, 44.60), (26.07, 46.36), (24.64, 46.73))
    poly += _cubic_bezier((24.64, 46.73), (23.76, 46.73), (23.09, 46.46), (22.63, 45.92))
    poly += _cubic_bezier((22.63, 45.92), (22.17, 45.38), (21.94, 44.68), (21.94, 43.81))
    poly += _cubic_bezier((21.94, 43.81), (21.94, 43.17), (22.12, 42.54), (22.48, 41.93))
    poly += _cubic_bezier((22.48, 41.93), (22.84, 41.31), (23.38, 40.42), (24.10, 39.25))
    poly += _cubic_bezier((24.10, 39.25), (24.83, 38.08), (25.38, 36.56), (25.75, 34.70))
    poly += _cubic_bezier((25.75, 34.70), (26.12, 32.83), (26.42, 30.22), (26.64, 26.86))
    poly.append((24.50, 26.86))
    poly += _cubic_bezier((24.50, 26.86), (23.22, 26.86), (22.21, 27.14), (21.46, 27.70))
    poly += _cubic_bezier((21.46, 27.70), (20.71, 28.26), (19.95, 29.18), (19.18, 30.46))
    poly.append((18.18, 29.96))
    poly += _cubic_bezier((18.18, 29.96), (19.04, 27.80), (20.11, 26.16), (21.39, 25.04))
    poly += _cubic_bezier((21.39, 25.04), (22.67, 23.92), (24.44, 23.09), (26.70, 23.09))
    return poly


def _point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    x0, y0 = poly[-1]
    for i in range(n):
        x1, y1 = poly[i]
        if (y1 > y) != (y0 > y) and x < (x0 - x1) * (y - y1) / (y0 - y1) + x1:
            inside = not inside
        x0, y0 = x1, y1
    return inside


def _in_rounded_rect(x, y, size=64.0, r=12.0):
    if x < 0 or x >= size or y < 0 or y >= size:
        return False
    cx = r if x < r else (size - r if x > size - r else x)
    cy = r if y < r else (size - r if y > size - r else y)
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def _render_icon_png(size, poly):
    """Render the 64x64 rounded-square π icon at `size`x`size` RGBA PNG with 4x4 supersampling."""
    bg_r, bg_g, bg_b = 0x2A, 0x78, 0xD6
    scale = 64.0 / size
    grid = (0.125, 0.375, 0.625, 0.875)
    raw = bytearray()
    for py in range(size):
        raw.append(0)  # PNG filter type 0 (None)
        for px in range(size):
            bg_hits = 0
            fg_hits = 0
            for sy in grid:
                vy = (py + sy) * scale
                for sx in grid:
                    vx = (px + sx) * scale
                    if _in_rounded_rect(vx, vy):
                        bg_hits += 1
                        if _point_in_poly(vx, vy, poly):
                            fg_hits += 1
            if bg_hits == 0:
                raw.extend((0, 0, 0, 0))
            else:
                alpha = round(bg_hits * 255 / 16)
                t = fg_hits / bg_hits
                r = round(bg_r * (1.0 - t) + 255 * t)
                g = round(bg_g * (1.0 - t) + 255 * t)
                b = round(bg_b * (1.0 - t) + 255 * t)
                raw.extend((r, g, b, alpha))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 9)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def favicons():
    """Write favicon.svg, favicon.ico (48x48) and apple-touch-icon.png (180x180) in docs/ for search engines."""
    pi_path = (
        "M47.62 23.09V26.86H41.36C41.22 28.60 41.05 31.15 40.85 34.50"
        "C40.65 37.84 40.55 39.89 40.55 40.66C40.55 41.80 40.82 42.59 41.35 43.02"
        "C41.87 43.46 42.50 43.67 43.24 43.67C44.03 43.67 44.71 43.46 45.28 43.05"
        "C45.85 42.63 46.50 41.94 47.25 40.96L48.10 41.80C47.22 43.39 46.18 44.61 44.99 45.46"
        "C43.80 46.31 42.36 46.73 40.68 46.73C38.92 46.73 37.65 46.23 36.88 45.23"
        "C36.11 44.23 35.72 42.63 35.72 40.42C35.72 39.46 35.85 37.39 36.11 34.21"
        "C36.37 31.02 36.57 28.57 36.71 26.86H28.88C28.54 33.43 28.11 38.29 27.58 41.45"
        "C27.05 44.60 26.07 46.36 24.64 46.73C23.76 46.73 23.09 46.46 22.63 45.92"
        "C22.17 45.38 21.94 44.68 21.94 43.81C21.94 43.17 22.12 42.54 22.48 41.93"
        "C22.84 41.31 23.38 40.42 24.10 39.25C24.83 38.08 25.38 36.56 25.75 34.70"
        "C26.12 32.83 26.42 30.22 26.64 26.86H24.50C23.22 26.86 22.21 27.14 21.46 27.70"
        "C20.71 28.26 19.95 29.18 19.18 30.46L18.18 29.96C19.04 27.80 20.11 26.16 21.39 25.04"
        "C22.67 23.92 24.44 23.09 26.70 23.09H47.62Z"
    )
    (DOCS / "favicon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">\n'
        '  <rect width="64" height="64" rx="12" fill="#2a78d6"/>\n'
        f'  <path fill="#ffffff" d="{pi_path}"/>\n'
        "</svg>\n"
    )
    poly = _pi_polygon()
    png48 = _render_icon_png(48, poly)
    ico_header = struct.pack("<HHH", 0, 1, 1)
    ico_entry = struct.pack("<BBBBHHII", 48, 48, 0, 0, 1, 32, len(png48), 22)
    (DOCS / "favicon.ico").write_bytes(ico_header + ico_entry + png48)
    (DOCS / "apple-touch-icon.png").write_bytes(_render_icon_png(180, poly))


def crawler_files():
    urls = [SITE + ("" if href == "./" else href) for href, _ in NAV]
    (DOCS / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {SITE}sitemap.xml\n")
    (DOCS / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <url><loc>{u}</loc></url>\n" for u in urls) + "</urlset>\n")
    raw = GITHUB.replace("github.com", "raw.githubusercontent.com") + "/main/"
    (DOCS / "llms.txt").write_text(f"""# {NAME}

> How weight quantization changes exact recall in Qwen 3.5 4B, Qwen 3.5 9B and Qwen 3.8 27B. 64 GGUF versions,
> from BF16 down to about 2 bits per weight, were run through llama.cpp and asked to recite π, e and √2 from
> memory, copy random digits from the prompt, and choose between prompt and memory when a planted digit
> disagrees with π. A teacher-forced pass records how sure each version was of every digit.

Main findings:

- Recall of π holds only at high bit widths, then declines. Q4_K_M (4.9 to 5.2 bits per weight) lost part of
  the first 1,000 digits on all three models. Full recall needed Q5_K_M on the 9B and 27B, and Q8_0 on the 4B.
- Recall stops on digits the uncompressed model was already least sure of.
- Copying digits from the prompt survives far more compression than recall does.
- Counted in bits per weight, bigger models take more compression.

This is a learning project. Claude Code ran the experiments and wrote the reports, and the author chose the
questions and reviewed the results. The conclusions are notes, not peer-reviewed research.

## Pages

- [Results]({SITE}): verdicts for hypotheses H1 to H6 on each model, with figures
- [How the experiments work]({SITE}method): the seven tests with real prompts, and teacher forcing explained
- [How the model files were made]({SITE}guide): Qwen 3.5 layers, quantization types, recipes and imatrix
- [Raw data]({SITE}data): every version's scores, every scored answer, per-digit certainty for π

## Reports (Markdown)

- [Phase 1, Qwen 3.5 4B]({raw}REPORT.md)
- [Phase 2, Qwen 3.5 9B]({raw}REPORT_9B.md)
- [Phase 3, Qwen 3.8 27B and the three-size comparison]({raw}REPORT_27B.md)
- [Follow-ups]({raw}REPORT_FOLLOWUP.md)
- [Test plan and hypotheses]({raw}PLAN.md)
- [Quantization guide]({raw}QUANTIZATION_GUIDE.md)

## Data

- [All versions, headline columns (JSON)]({SITE}downloads/versions.json)
- Per-model headline tables (CSV): """ + ", ".join(
            f"[{m}]({SITE}downloads/{m}_headline.csv)" for m in ("qwen3.5-4b", "qwen3.5-9b", "qwen3.8-27b")) + f"""
- [Source code and scored data]({GITHUB})
""")


def main():
    favicons()
    files = export_data()
    data_files = [p.name for p in (DOCS / "downloads").iterdir()]
    for page in sorted(DOCS.glob("*.html")):
        if page.name not in ("404.html", "guide.html"):
            set_topbar(page)
            seo(page, data_files)
    build_guide()
    seo(DOCS / "guide.html", data_files)
    not_found()
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
    crawler_files()
    (DOCS / ".nojekyll").touch()
    print(f"docs/: guide.html generated, {len(pages)} pages, {len(wanted)} figures copied, "
          f"{len(files)} data files")


if __name__ == "__main__":
    main()
