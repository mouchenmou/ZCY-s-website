from pathlib import Path
from urllib.parse import unquote
import re
import shutil


ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / "COMPSCI 182 homework 02.md"
BUILD = ROOT / "tmp" / "pdfs"
ASSETS = BUILD / "assets"
TEX = BUILD / "COMPSCI_182_homework_02.tex"


def escape_plain(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_escape(text: str) -> str:
    parts = re.split(r"(\$[^$]*\$)", text)
    escaped_parts = []
    for part in parts:
        if part.startswith("$") and part.endswith("$"):
            escaped_parts.append(part)
            continue
        code_parts = re.split(r"(`[^`]+`)", part)
        for code_part in code_parts:
            if code_part.startswith("`") and code_part.endswith("`"):
                escaped_parts.append(r"\texttt{" + escape_plain(code_part[1:-1]) + "}")
            else:
                escaped_parts.append(escape_plain(code_part))
    return "".join(escaped_parts)


def image_path(markdown_target: str, index: int) -> str:
    source = ROOT / unquote(markdown_target)
    if not source.exists():
        raise FileNotFoundError(f"Missing image: {source}")
    suffix = source.suffix.lower()
    destination = ASSETS / f"image_{index:03d}{suffix}"
    shutil.copy2(source, destination)
    return str(destination.relative_to(BUILD))


def flush_paragraph(lines, out):
    if not lines:
        return
    paragraph = " ".join(line.strip() for line in lines if line.strip())
    if paragraph:
        out.append(latex_escape(paragraph))
        out.append("")
    lines.clear()


def convert():
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.85in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{fontspec}",
        r"\usepackage{parskip}",
        r"\usepackage{hyperref}",
        r"\setmainfont{Times New Roman}",
        r"\setsansfont{Arial}",
        r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{6pt}",
        r"\graphicspath{{./}}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\LARGE COMPSCI 182 Homework 02}\\[8pt]",
        r"{\large Chenyang Zhang}",
        r"\end{center}",
        r"\vspace{0.2in}",
    ]
    paragraph = []
    image_index = 0

    for raw_line in MD.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        match = re.fullmatch(r"!\[[^\]]*\]\((.+)\)", stripped)
        if match:
            flush_paragraph(paragraph, out)
            image_index += 1
            asset = image_path(match.group(1), image_index)
            out.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=0.96\textwidth,height=0.78\textheight,keepaspectratio]{{{asset}}}",
                    r"\end{figure}",
                    "",
                ]
            )
            continue

        if stripped.startswith("## "):
            flush_paragraph(paragraph, out)
            out.append(r"\section*{" + latex_escape(stripped[3:]) + "}")
            continue

        if stripped.startswith("### "):
            flush_paragraph(paragraph, out)
            out.append(r"\subsection*{" + latex_escape(stripped[4:]) + "}")
            continue

        if stripped == "---":
            flush_paragraph(paragraph, out)
            out.append(r"\vspace{0.1in}\hrule\vspace{0.1in}")
            continue

        if not stripped:
            flush_paragraph(paragraph, out)
            continue

        paragraph.append(line)

    flush_paragraph(paragraph, out)
    out.append(r"\end{document}")
    TEX.write_text("\n".join(out), encoding="utf-8")
    print(TEX)


if __name__ == "__main__":
    convert()
