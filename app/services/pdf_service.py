"""PDF to Word conversion using PyMuPDF + python-docx.

Smart paragraph grouping: consecutive lines with similar font size,
Y-proximity, and indentation are merged into one paragraph.
Heading detection: text with font size >= 1.3x body size becomes a heading.

For scanned/image-based PDFs: Markdown (from OCR + LLM) → .docx.
"""

import io
import re
from collections import Counter

import fitz  # PyMuPDF
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement


async def get_pdf_page_count(pdf_file_bytes: bytes) -> int:
    """Get number of pages in PDF."""
    pdf_stream = io.BytesIO(pdf_file_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


def is_scanned_pdf(file_bytes: bytes) -> bool:
    """Check if any page in the PDF lacks a text layer (scanned/image-based).

    Returns True if ANY page has insufficient extractable text (< 20 chars),
    indicating a scanned/image-based PDF that needs OCR.
    """
    pdf_stream = io.BytesIO(file_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        # If any page has very little text, treat as scanned
        if len(text) < 20:
            doc.close()
            return True
    doc.close()
    return False


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


def markdown_to_docx(markdown_text: str) -> bytes:
    """Convert structured markdown (from LLM) to .docx bytes.

    Handles: headings (# ## ###), tables (|...|), bullet lists (- *),
    numbered lists (1. 2.), block quotes (>), and regular paragraphs.
    Inline formatting: **bold** and *italic* supported.
    """
    doc = Document()

    # Default style
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    lines = markdown_text.split("\n")
    i = 0
    in_table = False
    table_rows = []

    while i < len(lines):
        line = lines[i].rstrip()

        # Empty line: flush pending table, then skip
        if not line:
            if in_table and table_rows:
                _build_table(doc, table_rows)
                table_rows = []
                in_table = False
            i += 1
            continue

        # Table row detection
        if line.startswith("|") and line.endswith("|"):
            # Skip separator rows like |---|---|
            if re.match(r"^\|[\s\-:|]+\|$", line):
                in_table = True
                i += 1
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            table_rows.append(cells)
            in_table = True
            i += 1
            continue
        else:
            # Flush pending table
            if in_table and table_rows:
                _build_table(doc, table_rows)
                table_rows = []
                in_table = False

        # Heading: ###, ##, #
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            p = doc.add_paragraph()
            p.style = doc.styles[f"Heading {level}"]
            _add_inline_runs(p, text)
            i += 1
            continue

        # Bullet list: "- " or "* "
        if re.match(r"^[\-\*]\s+", line):
            text = re.sub(r"^[\-\*]\s+", "", line)
            p = doc.add_paragraph(style="List Bullet")
            _add_inline_runs(p, text)
            i += 1
            continue

        # Numbered list: "1. " etc.
        num_match = re.match(r"^(\d+)\.\s+(.+)$", line)
        if num_match:
            p = doc.add_paragraph(style="List Number")
            _add_inline_runs(p, num_match.group(2))
            i += 1
            continue

        # Block quote: "> "
        if line.startswith("> "):
            text = line[2:]
            p = doc.add_paragraph()
            _add_inline_runs(p, text)
            p.paragraph_format.left_indent = Inches(0.5)
            _set_paragraph_spacing(p, space_before=0, space_after=6)
            i += 1
            continue

        # Regular paragraph
        _add_inline_runs(doc.add_paragraph(), line)
        i += 1

    # Flush remaining table
    if in_table and table_rows:
        _build_table(doc, table_rows)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


def _add_inline_runs(paragraph, text: str):
    """Add text with **bold** and *italic* markdown parsing to a paragraph."""
    pattern = re.compile(r"(\*\*(.+?)\*\*|\*(.+?)\*)")
    last_end = 0
    for match in pattern.finditer(text):
        if match.start() > last_end:
            paragraph.add_run(text[last_end:match.start()])
        if match.group(2):  # **bold**
            run = paragraph.add_run(match.group(2))
            run.bold = True
        elif match.group(3):  # *italic*
            run = paragraph.add_run(match.group(3))
            run.italic = True
        last_end = match.end()
    if last_end < len(text):
        paragraph.add_run(text[last_end:])
    _set_paragraph_spacing(paragraph, space_before=0, space_after=6)


def _build_table(doc, rows):
    """Build a docx table from a list of cell lists."""
    if not rows:
        return
    num_cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=num_cols, style="Table Grid")
    for r_idx, row in enumerate(rows):
        for c_idx, cell_text in enumerate(row):
            if c_idx < num_cols:
                cell = table.cell(r_idx, c_idx)
                cell.text = ""
                p = cell.paragraphs[0]
                _add_inline_runs(p, cell_text)
                # Bold first row as header
                if r_idx == 0:
                    for run in p.runs:
                        run.bold = True
    doc.add_paragraph()  # spacing after table


async def convert_scanned_pdf_to_word(
    pdf_file_bytes: bytes,
    filename: str,
    ocr_markdown: str,
    restructured_markdown: str,
) -> bytes:
    """Convert scanned PDF to Word using OCR + LLM restructured markdown.

    Args:
        pdf_file_bytes: Original PDF bytes (for page count info)
        filename: Original filename
        ocr_markdown: Raw markdown from OCR model (datalab-to/marker)
        restructured_markdown: Cleaned markdown from Llama 3.1 405B

    Returns .docx bytes built from the restructured markdown.
    """
    # Use the restructured markdown if non-empty, otherwise fallback to raw OCR
    source = restructured_markdown if restructured_markdown.strip() else ocr_markdown
    return markdown_to_docx(source)
