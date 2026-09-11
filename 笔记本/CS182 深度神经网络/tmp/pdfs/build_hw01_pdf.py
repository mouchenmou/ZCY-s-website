from pathlib import Path
from urllib.parse import unquote
import re

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    HRFlowable,
)


ROOT = Path(__file__).resolve().parents[2]
HW_DIR = ROOT / "Homework" / "HW01"
MD_PATH = HW_DIR / "COMPSCI 182 homework 01.md"
COLAB_PDF = HW_DIR / "q_relu_vis.ipynb - Colab.pdf"
OUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs"
HOMEWORK_PDF = TMP_DIR / "COMPSCI 182 homework 01.pdf"
FINAL_PDF = OUT_DIR / "COMPSCI 182 homework 01 combined.pdf"


def resolve_image(raw: str) -> Path:
    raw = raw.strip()
    if raw.startswith("<") and raw.endswith(">"):
        raw = raw[1:-1]
    raw = unquote(raw)
    return (HW_DIR / raw).resolve()


def fit_image(path: Path, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    width, height = image.imageWidth, image.imageHeight
    scale = min(max_width / width, max_height / height, 1.0)
    image.drawWidth = width * scale
    image.drawHeight = height * scale
    image.hAlign = "CENTER"
    return image


def build_homework_pdf() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        spaceBefore=12,
        spaceAfter=7,
    )
    h3 = ParagraphStyle(
        "H3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        spaceBefore=8,
        spaceAfter=5,
    )

    doc = SimpleDocTemplate(
        str(HOMEWORK_PDF),
        pagesize=A4,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.58 * inch,
        title="COMPSCI 182 homework 01",
        author="Chenyang Zhang",
    )

    story = []
    image_re = re.compile(r"!\[[^\]]*\]\((.+)\)")
    lines = MD_PATH.read_text(encoding="utf-8").splitlines()
    paragraph = []

    def flush_paragraph():
        if paragraph:
            text = " ".join(paragraph).strip()
            story.append(Paragraph(text, body))
            paragraph.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            story.append(Spacer(1, 4))
            continue
        if stripped == "---":
            flush_paragraph()
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#9CA3AF")))
            story.append(PageBreak())
            continue
        image_match = image_re.match(stripped)
        if image_match:
            flush_paragraph()
            image_path = resolve_image(image_match.group(1))
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            story.append(fit_image(image_path, doc.width, 6.65 * inch))
            story.append(Spacer(1, 8))
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(stripped[4:], h3))
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(stripped[3:], h2))
            continue
        paragraph.append(stripped)

    flush_paragraph()
    doc.build(story)


def merge_pdfs() -> None:
    writer = PdfWriter()
    for pdf_path in (HOMEWORK_PDF, COLAB_PDF):
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            writer.add_page(page)
    with FINAL_PDF.open("wb") as handle:
        writer.write(handle)


if __name__ == "__main__":
    build_homework_pdf()
    merge_pdfs()
    print(FINAL_PDF)
