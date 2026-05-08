"""PDF to Word conversion service."""

import io

import fitz  # PyMuPDF


async def get_pdf_page_count(pdf_file_bytes: bytes) -> int:
    """Get number of pages in PDF using PyMuPDF."""
    pdf_stream = io.BytesIO(pdf_file_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    page_count = len(doc)
    doc.close()
    return page_count


# TODO: Implement PDF to Word conversion (pdf2docx has system deps,
# will use LibreOffice or cloud API in production)
async def convert_pdf_to_word(pdf_file_bytes: bytes, filename: str) -> bytes:
    """Placeholder - will be implemented with LibreOffice or cloud API."""
    raise NotImplementedError("PDF to Word conversion not yet implemented")
