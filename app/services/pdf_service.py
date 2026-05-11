"""PDF to Word conversion using PyMuPDF + python-docx."""

import io

import fitz  # PyMuPDF
from docx import Document
from docx.shared import Pt, Inches


async def get_pdf_page_count(pdf_file_bytes: bytes) -> int:
    """Get number of pages in PDF."""
    pdf_stream = io.BytesIO(pdf_file_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


async def convert_pdf_to_word(pdf_file_bytes: bytes, filename: str) -> bytes:
    """Convert PDF to DOCX using PyMuPDF text extraction.

    Tries 3 extraction methods in order:
    1. Structured dict extraction (preserves font size/bold)
    2. Plain text fallback
    3. Word-level extraction as last resort

    Raises ValueError if no text can be extracted (e.g. scanned/image PDF).
    """
    pdf_stream = io.BytesIO(pdf_file_bytes)
    pdf = fitz.open(stream=pdf_stream, filetype="pdf")
    docx = Document()

    style = docx.styles["Normal"]
    font = style.font
    font.name = "Arial"
    font.size = Pt(11)

    total_paragraphs = 0

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        has_content = False

        # Method 1: structured dict extraction (preserves formatting)
        try:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        text = "".join([span["text"] for span in line["spans"]])
                        if text.strip():
                            first_span = line["spans"][0]
                            font_size = first_span.get("size", 11)
                            is_bold = "Bold" in first_span.get("font", "")

                            p = docx.add_paragraph()
                            run = p.add_run(text)
                            run.font.size = Pt(font_size)
                            run.bold = is_bold
                            has_content = True

                elif block["type"] == 1:  # Image block
                    for img_info in block.get("images", []):
                        try:
                            docx.add_picture(
                                io.BytesIO(img_info["image"]), width=Inches(5)
                            )
                            has_content = True
                        except Exception:
                            pass
        except Exception:
            pass

        # Method 2: fallback plain text if dict extraction found nothing
        if not has_content:
            plain_text = page.get_text("text")
            if plain_text.strip():
                for line in plain_text.split("\n"):
                    stripped = line.strip()
                    if stripped:
                        docx.add_paragraph(stripped)
                has_content = True

        # Method 3: extract as raw words if still nothing
        if not has_content:
            words = page.get_text("words")
            if words:
                text = " ".join([w[4] for w in words if len(w) > 4 and w[4].strip()])
                if text.strip():
                    docx.add_paragraph(text)
                    has_content = True

        if page_num < len(pdf) - 1:
            docx.add_page_break()

        total_paragraphs += len(docx.paragraphs)

    pdf.close()

    # Verify content was actually extracted
    if total_paragraphs == 0:
        raise ValueError(
            "This PDF appears to be a scanned document or image without a text layer. "
            "PDF to Word conversion requires PDFs with selectable text. "
            "Please use an OCR tool first to add a text layer to your PDF."
        )

    buf = io.BytesIO()
    docx.save(buf)
    buf.seek(0)
    return buf.read()
