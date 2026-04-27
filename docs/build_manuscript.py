"""
Build submission-ready Word manuscript from paper_draft_v3.md.
Embeds figures inline where referenced. Applies journal-appropriate formatting.

Run: python docs/build_manuscript.py
Output: docs/NeoResist_MD_manuscript.docx
"""
from __future__ import annotations
import re
import subprocess
import tempfile
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
DRAFT  = ROOT / "docs" / "paper_draft_v3.md"
FIGS   = ROOT / "figures"
OUT    = ROOT / "docs" / "NeoResist_MD_manuscript.docx"
PANDOC = Path("C:/Users/rambe/AppData/Local/Pandoc/pandoc.exe")

def inject_figures(md: str) -> str:
    """Replace Figure N references with pandoc-style image links."""
    fig_map = {
        "Figure 1": "fig1_framework.png",
        "Figure 2A": "fig2_melanoma.png",
        "Figure 2B": "fig2_melanoma.png",
        "Figure 2":  "fig2_melanoma.png",
        "Figure 3A": "fig3_borch.png",
        "Figure 3B": "fig3_borch.png",
        "Figure 3":  "fig3_borch.png",
        "Figure 4":  "fig4_transfer.png",
    }
    # Insert figures at Figure Legend section
    legends = {
        "**Figure 1.": "fig1_framework.png",
        "**Figure 2.": "fig2_melanoma.png",
        "**Figure 3.": "fig3_borch.png",
        "**Figure 4.": "fig4_transfer.png",
    }
    lines = md.split("\n")
    out = []
    for line in lines:
        out.append(line)
        for marker, fname in legends.items():
            if line.startswith(marker):
                img_path = (FIGS / fname).as_posix()
                out.append(f"\n![]({img_path}){{width=100%}}\n")
    return "\n".join(out)

def build() -> None:
    md = DRAFT.read_text(encoding="utf-8")
    md_with_figs = inject_figures(md)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md",
                                     encoding="utf-8", delete=False) as f:
        f.write(md_with_figs)
        tmp = Path(f.name)

    cmd = [
        str(PANDOC),
        str(tmp),
        "-o", str(OUT),
        "--from", "markdown",
        "--to", "docx",
        "--highlight-style", "tango",
        "-V", "geometry:margin=2.5cm",
        "--toc-depth=2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    tmp.unlink()

    if result.returncode != 0:
        print("ERROR:", result.stderr)
    else:
        size_kb = OUT.stat().st_size // 1024
        print(f"Created: {OUT}  ({size_kb} KB)")

if __name__ == "__main__":
    build()
