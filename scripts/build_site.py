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

import pyarrow.parquet as pq

from common import FIGURES, ROOT, SCORES, TF, load_config, read_jsonl

DOCS = ROOT / "docs"
SITE = "https://bits-of-pi.engineeringthefuture.ai/"
GITHUB = "https://github.com/engineeringthefutureai/bits-of-pi"
REPO = GITHUB + "/blob/main/"
NAME = "Qwen Digit Recall"
SHARE_IMAGE = "H5_model_size.png"
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
    site = {"@type": "WebSite", "@id": SITE + "#site", "name": NAME, "url": SITE}
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
                "image": SITE + "figures/" + SHARE_IMAGE}
    return json.dumps({"@context": "https://schema.org", "@graph": [site, item]}, ensure_ascii=False, indent=1)


def seo(page, files):
    """Rewrite one page's search-engine block in <head> and give every image its size."""
    text = page.read_text()
    text = re.sub(r"<!-- seo -->.*?<!-- /seo -->\n", "", text, flags=re.S)
    title = html.unescape(re.search(r"<title>([^<]*)</title>", text)[1])
    description = html.unescape(re.search(r'<meta name="description" content="([^"]*)">', text)[1])
    author = re.search(r'<meta name="author" content="([^"]*)">', text)[1]
    w, h = png_size(FIGURES / SHARE_IMAGE)
    block = (f'<!-- seo -->\n<link rel="canonical" href="{page_url(page.name)}">\n'
             f'<meta property="og:site_name" content="{NAME}">\n'
             f'<meta property="og:image:width" content="{w}">\n<meta property="og:image:height" content="{h}">\n'
             '<meta property="og:image:alt" content="π digits recalled against bits per weight for Qwen 3.5 4B, '
             '9B and Qwen 3.8 27B">\n'
             '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
             '<script type="application/ld+json">\n'
             + structured_data(page.name, title, description, author, files) + "\n</script>\n<!-- /seo -->\n")
    text = text.replace("</head>", block + "</head>", 1)

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
    build_guide()
    not_found()
    files = export_data()
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
    data_files = [p.name for p in (DOCS / "downloads").iterdir()]
    for page in sorted(DOCS.glob("*.html")):
        if page.name not in ("404.html", "guide.html"):
            set_topbar(page)
        if page.name != "404.html":
            seo(page, data_files)
    crawler_files()
    (DOCS / ".nojekyll").touch()
    print(f"docs/: guide.html generated, {len(pages)} pages, {len(wanted)} figures copied, "
          f"{len(files)} data files")


if __name__ == "__main__":
    main()
