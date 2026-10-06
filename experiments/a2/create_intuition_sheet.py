"""Render the A2 intuition-sheet Markdown as a printable PDF."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from experiments.a2.helpers.paths import OUTPUT_ROOT


SOURCE = OUTPUT_ROOT / "intuition_sheet.md"
TARGET = OUTPUT_ROOT / "intuition_sheet_printable.pdf"
DETAILED_TARGET = OUTPUT_ROOT / "intuition_sheet_problem_by_problem.pdf"
PAGE_SIZE = (8.5, 11)
LEFT = 0.075
RIGHT = 0.94
TOP = 0.94
BOTTOM = 0.065


def _plain(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    replacements = {
        "ηλ": "eta*lambda",
        "μ": "mu",
        "η": "eta",
        "λ": "lambda",
        "σ": "sigma",
        "α": "alpha",
        "β": "beta",
        "ω": "omega",
        "ε": "epsilon",
        "Θ": "Theta",
        "Δ": "Delta",
        "²": "^2",
        "−": "-",
        "–": "-",
        "—": "-",
        "×": "x",
        "→": "->",
        "∥": "||",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def _wrap(text: str, width: int, prefix: str = "") -> list[str]:
    return textwrap.wrap(
        _plain(text),
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    ) or [prefix.rstrip()]


def _blocks(markdown: str, *, include_all_cited_graphs: bool = False):
    lines = markdown.splitlines()
    cited_graphs = set(re.findall(r"`(plots/[^`]+)`", markdown))
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue
        image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if image_match:
            stem = re.sub(r"\.(png|pdf)$", "", image_match.group(2))
            if include_all_cited_graphs and stem in cited_graphs:
                index += 1
                continue
            yield "image", [image_match.group(1), image_match.group(2)]
            index += 1
            continue
        if line.strip() == "<!-- pagebreak -->":
            yield "pagebreak", []
            index += 1
            continue
        if include_all_cited_graphs and line.startswith(("**Graph:**", "**Graphs:**")):
            yield "paragraph", [line]
            for citation in re.findall(r"`(plots/[^`]+)`", line):
                yield "image", [f"Cited graph: {Path(citation).name}", f"{citation}.png"]
            index += 1
            continue
        if line.startswith("|"):
            table = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                candidate = lines[index].strip()
                if not re.fullmatch(r"\|?[\s:|-]+\|?", candidate):
                    table.append(candidate)
                index += 1
            yield "table", table
            continue
        if line.startswith("# "):
            yield "title", [line[2:]]
        elif line.startswith("## "):
            yield "heading", [line[3:]]
        elif line.startswith("### "):
            yield "subheading", [line[4:]]
        elif line.startswith("- "):
            yield "bullet", [line[2:]]
        elif re.match(r"^\d+\. ", line):
            marker, text = line.split(" ", 1)
            yield "number", [marker, text]
        else:
            paragraph = [line]
            index += 1
            while index < len(lines):
                following = lines[index].rstrip()
                if (
                    not following
                    or following.startswith(("#", "- ", "|"))
                    or re.match(r"^\d+\. ", following)
                ):
                    break
                paragraph.append(following)
                index += 1
            yield "paragraph", [" ".join(paragraph)]
            continue
        index += 1


def _validate_graph_citations(markdown: str, source_root: Path) -> None:
    missing = []
    for citation in sorted(set(re.findall(r"`(plots/[^`]+)`", markdown))):
        path = source_root / citation
        candidates = (path, Path(f"{path}.png"))
        if not any(candidate.is_file() for candidate in candidates):
            missing.append(citation)
    if missing:
        raise FileNotFoundError(f"Missing cited graph bundles: {missing}")


class Renderer:
    def __init__(self, pdf: PdfPages, source_root: Path):
        self.pdf = pdf
        self.source_root = source_root
        self.page = 0
        self.figure = None
        self.axis = None
        self.y = TOP
        self.images = 0
        self.new_page()

    def new_page(self):
        if self.figure is not None:
            self.finish_page()
        self.page += 1
        self.figure, self.axis = plt.subplots(figsize=PAGE_SIZE)
        self.axis.set_axis_off()
        self.axis.set_xlim(0, 1)
        self.axis.set_ylim(0, 1)
        self.y = TOP
        self.axis.text(
            LEFT,
            0.975,
            "CS 312 · Assignment 2 · Hyperparameter Scaling",
            fontsize=8,
            color="#555555",
            va="top",
        )

    def finish_page(self):
        self.axis.text(
            RIGHT,
            0.025,
            f"{self.page}",
            fontsize=8,
            color="#666666",
            va="bottom",
            ha="right",
        )
        self.pdf.savefig(self.figure, bbox_inches="tight", pad_inches=0.15)
        plt.close(self.figure)
        self.figure = None

    def ensure(self, height: float):
        if self.y - height < BOTTOM:
            self.new_page()

    def text(self, lines, *, size=8.5, weight="normal", color="#222222", indent=0.0, gap=0.007):
        line_height = 0.0175 if size <= 9.5 else 0.022
        height = line_height * len(lines) + gap
        self.ensure(height)
        for line in lines:
            self.axis.text(
                LEFT + indent,
                self.y,
                line,
                fontsize=size,
                fontweight=weight,
                color=color,
                va="top",
                ha="left",
                family="DejaVu Sans",
            )
            self.y -= line_height
        self.y -= gap

    def image(self, caption: str, source: str):
        path = (self.source_root / source).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing cited graph: {path}")
        pixels = plt.imread(path)
        height_px, width_px = pixels.shape[:2]
        width = RIGHT - LEFT
        height = min(0.235, width * (height_px / width_px) * (PAGE_SIZE[0] / PAGE_SIZE[1]))
        caption_lines = _wrap(f"Figure — {caption} [{source}]", 105)
        caption_height = 0.016 * len(caption_lines) + 0.012
        self.ensure(height + caption_height + 0.012)
        bottom = self.y - height
        self.axis.imshow(
            pixels,
            extent=(LEFT, RIGHT, bottom, self.y),
            interpolation="antialiased",
            aspect="auto",
            zorder=1,
        )
        self.images += 1
        self.axis.set_xlim(0, 1)
        self.axis.set_ylim(0, 1)
        self.y = bottom - 0.007
        self.text(caption_lines, size=7.3, color="#555555", indent=0.006, gap=0.010)

    def render(self, kind: str, values: list[str]):
        if kind == "title":
            self.text(_wrap(values[0], 58), size=18, weight="bold", color="#17365d", gap=0.018)
        elif kind == "heading":
            self.y -= 0.004
            self.text(_wrap(values[0], 78), size=11.5, weight="bold", color="#17365d", gap=0.010)
        elif kind == "subheading":
            self.text(_wrap(values[0], 90), size=9.8, weight="bold", color="#2f5597", gap=0.006)
        elif kind == "image":
            self.image(values[0], values[1])
        elif kind == "pagebreak":
            if self.y < TOP - 0.01:
                self.new_page()
        elif kind == "bullet":
            self.text(_wrap(values[0], 104, "• "), indent=0.015, gap=0.003)
        elif kind == "number":
            self.text(_wrap(values[1], 101, f"{values[0]} "), indent=0.015, gap=0.003)
        elif kind == "table":
            rows = [[_plain(cell.strip()) for cell in row.strip("|").split("|")] for row in values]
            if not rows:
                return
            widths = [max(len(row[i]) if i < len(row) else 0 for row in rows) for i in range(len(rows[0]))]
            total = max(1, sum(widths))
            widths = [max(8, int(width * 100 / total)) for width in widths]
            rendered = []
            for row_index, row in enumerate(rows):
                cells = []
                for i, width in enumerate(widths):
                    cell = row[i] if i < len(row) else ""
                    cells.append(cell[:width].ljust(width))
                rendered.append(" | ".join(cells))
                if row_index == 0:
                    rendered.append("-" * min(118, len(rendered[-1])))
            self.text(rendered, size=7.0, color="#333333", indent=0.008, gap=0.010)
        else:
            self.text(_wrap(values[0], 108), gap=0.007)


def generate(
    source: Path = SOURCE,
    target: Path = TARGET,
    *,
    include_all_cited_graphs: bool = False,
) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    markdown = source.read_text()
    _validate_graph_citations(markdown, source.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(target) as pdf:
        renderer = Renderer(pdf, source.parent)
        for kind, values in _blocks(markdown, include_all_cited_graphs=include_all_cited_graphs):
            renderer.render(kind, values)
        renderer.finish_page()
        print(f"Rendered {renderer.page} pages with {renderer.images} embedded figures")
    return target


def main() -> int:
    compact = generate()
    print(f"Wrote {compact}")
    detailed = generate(target=DETAILED_TARGET, include_all_cited_graphs=True)
    print(f"Wrote {detailed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
