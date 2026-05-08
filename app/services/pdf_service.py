"""PDF to Word conversion service."""

import io
import os
import tempfile

from pdf2docx import Converter


async def convert_pdf_to_word(pdf_file_bytes: bytes, filename: str) -> bytes:
    """Convert PDF bytes to DOCX bytes using pdf2docx."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_tmp:
        pdf_tmp.write(pdf_file_bytes)
        pdf_tmp.flush()

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as docx_tmp:
            docx_tmp.close()

            cv = Converter(pdf_tmp.name)
            cv.convert(docx_tmp.name)
            cv.close()

            with open(docx_tmp.name, "rb") as f:
                result = f.read()

            os.unlink(docx_tmp.name)

        os.unlink(pdf_tmp.name)

    return result


async def get_pdf_page_count(pdf_file_bytes: bytes) -> int:
    """Get number of pages in PDF using pdf2docx's underlying fitz."""
    import fitz

    pdf_stream = io.BytesIO(pdf_file_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    page_count = len(doc)
    doc.close()
    return page_count
