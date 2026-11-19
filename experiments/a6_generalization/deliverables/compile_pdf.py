"""Render a deliverable markdown file to PDF with pandoc and WeasyPrint.

WeasyPrint on this Mac needs Homebrew's Pango libraries, and it does not draw
MathML superscripts. LaTeX in the markdown is turned into HTML before pandoc.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOMEBREW_LIB = '/opt/homebrew/lib'


def _ensure_libraries():
    current = os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')
    if HOMEBREW_LIB in current.split(':'):
        return
    env = os.environ.copy()
    env['DYLD_FALLBACK_LIBRARY_PATH'] = HOMEBREW_LIB + ((':' + current) if current else '')
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def _tex(source):
    text = source.strip()
    text = text.replace(r'\rho', 'ρ').replace(r'\in', ' ∈ ')
    text = text.replace(r'\{', '').replace(r'\}', '')
    text = text.replace(r'\mathrm{LR}', 'LR').replace(r'\mathrm{WD}', 'WD')
    text = text.replace(r'\times', '×').replace(r'\,', ' ').replace(r'\ldots', '…')
    text = text.replace(r'\log', 'log').replace(r'\approx', '≈').replace(r'\le', '≤')
    text = text.replace(r'\beta', 'β').replace(r'\hat L', 'L̂').replace(r'\hat{L}', 'L̂')
    text = re.sub(r'\\sqrt\{([^{}]+)\}', r'√(\1)', text)

    def _sup(match):
        return f'<sup>{match.group(1) or match.group(2)}</sup>'

    def _sub(match):
        return f'<sub>{match.group(1) or match.group(2)}</sub>'

    text = re.sub(r'\^\{([^{}]+)\}|\^([A-Za-z0-9.+-]+)', _sup, text)
    text = re.sub(r'_\{([^{}]+)\}|_([A-Za-z0-9.+-]+)', _sub, text)
    text = text.replace('{', '').replace('}', '').replace(r'\ ', ' ')
    return text


def _math_to_html(markdown):
    def display(match):
        return '\n\n<div class="equation">' + _tex(match.group(1)) + '</div>\n\n'

    def inline(match):
        return _tex(match.group(1))

    markdown = re.sub(r'\\\[(.*?)\\\]', display, markdown, flags=re.S)
    markdown = re.sub(r'\\\((.*?)\\\)', inline, markdown, flags=re.S)
    # Pandoc reads a leading "(A)" as an alphabetical list and drops the label.
    markdown = re.sub(r'^( *)- \(([A-D])\) ', r'\1- **(\2)** ', markdown, flags=re.M)
    return markdown


def main():
    _ensure_libraries()
    from weasyprint import HTML

    src = Path(sys.argv[1]).resolve()
    dst = Path(sys.argv[2]).resolve()
    prepared = _math_to_html(src.read_text())
    body = subprocess.check_output(
        ['pandoc', '-f', 'markdown', '-t', 'html'],
        input=prepared,
        text=True,
    )
    css = (HERE / 'pdf-style.css').read_text()
    html = (
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        + css
        + '</style></head><body>'
        + body
        + '</body></html>'
    )
    HTML(string=html, base_url=str(src.parent)).write_pdf(dst)
    print(dst)


if __name__ == '__main__':
    main()
