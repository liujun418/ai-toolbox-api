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
    """Convert PDF to DOCX using PyMuPDF text extraction."""
    pdf_stream = io.BytesIO(pdf_file_bytes)
    pdf = fitz.open(stream=pdf_stream, filetype="pdf")
    docx = Document()

    style = docx.styles["Normal"]
    font = style.font
    font.name = "Arial"
    font.size = Pt(11)

    for page_num in range(len(pdf)):
        page = pdf[page_num]

        # Extract text blocks with position info
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    text = "".join([span["text"] for span in line["spans"]])
                    if text.strip():
                        # Detect bold from first span
                        first_span = line["spans"][0]
                        font_size = first_span.get("size", 11)
                        is_bold = "Bold" in first_span.get("font", "")

                        p = docx.add_paragraph()
                        run = p.add_run(text)
                        run.font.size = Pt(font_size)
                        run.bold = is_bold

            elif block["type"] == 1:  # Image block
                # Extract and embed images
                for img_info in block.get("images", []):
                    try:
                        img_bytes = img_info["image"]
                        docx.add_picture(io.BytesIO(img_bytes), width=Inches(5))
                    except Exception:
                        pass

        if page_num < len(pdf) - 1:
            docx.add_page_break()

    pdf.close()

    buf = io.BytesIO()
    docx.save(buf)
    buf.seek(0)
    return buf.read()
