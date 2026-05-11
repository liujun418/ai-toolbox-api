"""PDF to Word conversion using PyMuPDF + python-docx.

Smart paragraph grouping: consecutive lines with similar font size,
Y-proximity, and indentation are merged into one paragraph.
Heading detection: text with font size >= 1.3x body size becomes a heading.
"""

import io
from collections import Counter

import fitz  # PyMuPDF
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.oxml.ns import qn


async def get_pdf_page_count(pdf_file_bytes: bytes) -> int:
    """Get number of pages in PDF."""
    pdf_stream = io.BytesIO(pdf_file_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


def _estimate_body_font_size(all_lines: list) -> float:
    """Find the most common font size (mode), which is usually the body text size."""
    sizes = []
    for line_info in all_lines:
        sizes.append(round(line_info["font_size"]))
    if not sizes:
        return 11.0
    counter = Counter(sizes)
    return float(counter.most_common(1)[0][0])


def _extract_lines_from_page(page) -> list[dict]:
    """Extract text lines from a page with position and formatting metadata."""
    lines = []
    try:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                text = "".join([span["text"] for span in line["spans"]])
                if not text.strip():
                    continue
                first_span = line["spans"][0]
                font_size = first_span.get("size", 11)
                font_name = first_span.get("font", "")
                is_bold = "Bold" in font_name
                is_italic = "Italic" in font_name or "Oblique" in font_name
                bbox = line.get("bbox", (0, 0, 0, 0))
                lines.append({
                    "text": text.strip(),
                    "font_size": font_size,
                    "font_name": font_name,
                    "is_bold": is_bold,
                    "is_italic": is_italic,
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                })
    except Exception:
        pass
    return lines


def _group_lines_into_paragraphs(lines: list, body_font_size: float) -> list[dict]:
    """Group consecutive lines into paragraphs.

    Two consecutive lines are in the same paragraph when ALL conditions hold:
    1. Vertical gap <= 1.2 * actual line height (measured from previous pair)
    2. Same font size (or within 1pt for body text)
    3. Similar indentation (x0 within 5pt)

    A heading breaks from body text when font size is significantly larger.
    A gap larger than the typical line spacing also starts a new paragraph.
    """
    if not lines:
        return []

    paragraphs = []
    current_lines = [lines[0]]

    for i in range(1, len(lines)):
        prev = lines[i - 1]
        curr = lines[i]

        vertical_gap = curr["y0"] - prev["y1"]
        font_diff = abs(curr["font_size"] - prev["font_size"])
        indent_diff = abs(curr["x0"] - prev["x0"])

        # Calculate typical line height from what we've seen so far
        if len(current_lines) >= 2:
            typical_line_h = current_lines[-1]["y1"] - current_lines[-1]["y0"]
        else:
            typical_line_h = body_font_size * 1.2

        gap_threshold = typical_line_h * 1.2

        # New paragraph if any condition fails
        new_paragraph = (
            vertical_gap > gap_threshold       # gap too large
            or font_diff > body_font_size * 0.15  # font size changed significantly
            or indent_diff > 5                   # indentation changed
        )

        if new_paragraph:
            paragraphs.append({"lines": current_lines})
            current_lines = [curr]
        else:
            current_lines.append(curr)

    if current_lines:
        paragraphs.append({"lines": current_lines})

    return paragraphs


def _apply_run_styling(run, line_info: dict):
    """Apply font styling to a run."""
    run.font.size = Pt(line_info["font_size"])
    run.bold = line_info["is_bold"]
    run.italic = line_info["is_italic"]
    if line_info["font_name"]:
        run.font.name = line_info["font_name"]
        rPr = run._element.get_or_add_rPr()
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = rPr.makeelement(qn("w:rFonts"), {})
            rPr.append(rFonts)
        rFonts.set(qn("w:ascii"), line_info["font_name"])
        rFonts.set(qn("w:hAnsi"), line_info["font_name"])


def _set_paragraph_spacing(p, space_before=0, space_after=6):
    """Set paragraph spacing in twips."""
    pPr = p._element.get_or_add_pPr()
    spacing = pPr.find(qn("w:spacing"))
    if spacing is None:
        spacing = pPr.makeelement(qn("w:spacing"), {})
        pPr.append(spacing)
    spacing.set(qn("w:before"), str(int(space_before * 20)))
    spacing.set(qn("w:after"), str(int(space_after * 20)))


def _add_formatted_paragraph(docx, lines: list, body_font_size: float):
    """Add a paragraph to the docx with proper formatting."""
    is_heading = any(l["font_size"] >= body_font_size * 1.3 for l in lines)

    p = docx.add_paragraph()

    if is_heading:
        heading_size = max(l["font_size"] for l in lines)
        heading_size = min(heading_size, 28)
        run = p.add_run(" ".join(l["text"] for l in lines))
        run.font.size = Pt(heading_size)
        run.bold = True
        _set_paragraph_spacing(p, space_before=12, space_after=6)
    else:
        for i, line_info in enumerate(lines):
            separator = "" if i == 0 else " "
            run = p.add_run(separator + line_info["text"])
            _apply_run_styling(run, line_info)
        _set_paragraph_spacing(p, space_before=0, space_after=6)


async def convert_pdf_to_word(pdf_file_bytes: bytes, filename: str) -> bytes:
    """Convert PDF to DOCX using PyMuPDF text extraction with smart grouping.

    Raises ValueError if no text can be extracted (e.g. scanned/image PDF).
    """
    pdf_stream = io.BytesIO(pdf_file_bytes)
    pdf = fitz.open(stream=pdf_stream, filetype="pdf")

    # Pass 1: collect all lines
    all_pages_lines = []
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        page_lines = _extract_lines_from_page(page)
        all_pages_lines.append(page_lines)

    # Estimate body font size
    all_lines_flat = [line for page_lines in all_pages_lines for line in page_lines]
    body_font_size = _estimate_body_font_size(all_lines_flat)

    docx = Document()

    # Default style
    style = docx.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(body_font_size)
    style.paragraph_format.space_before = Pt(0)
    style.paragraph_format.space_after = Pt(0)

    total_paragraphs = 0

    for page_num, page_lines in enumerate(all_pages_lines):
        paragraphs = _group_lines_into_paragraphs(page_lines, body_font_size)

        for pg in paragraphs:
            _add_formatted_paragraph(docx, pg["lines"], body_font_size)
            total_paragraphs += 1

        if page_num < len(all_pages_lines) - 1:
            docx.add_page_break()

    pdf.close()

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
