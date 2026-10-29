"""Refresh the GitHub Pages site in docs/: copy every figure docs/index.html references.

usage: build_site.py   (run after analyze.py; docs/index.html itself is edited by hand)
"""

import re
import shutil
import sys

from common import FIGURES, ROOT

DOCS = ROOT / "docs"


def main():
    page = (DOCS / "index.html").read_text()
    wanted = sorted(set(re.findall(r'src="figures/([^"]+)"', page)))
    missing = [f for f in wanted if not (FIGURES / f).exists()]
    if missing:
        sys.exit(f"figures referenced by docs/index.html but missing from figures/: {missing}")
    out = DOCS / "figures"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for f in wanted:
        shutil.copy2(FIGURES / f, out / f)
    (DOCS / ".nojekyll").touch()
    print(f"docs/: {len(wanted)} figures copied")


if __name__ == "__main__":
    main()
