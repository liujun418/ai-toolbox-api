"""PDF to Word conversion service using LibreOffice headless."""

import asyncio
import io
import os
import shutil
import tempfile

import fitz  # PyMuPDF


async def get_pdf_page_count(pdf_file_bytes: bytes) -> int:
    """Get number of pages in PDF using PyMuPDF."""
    pdf_stream = io.BytesIO(pdf_file_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    page_count = len(doc)
    doc.close()
    return page_count


async def convert_pdf_to_word(pdf_file_bytes: bytes, filename: str) -> bytes:
    """Convert PDF to DOCX using LibreOffice headless."""
    suffix = ".pdf"
    out_dir = tempfile.mkdtemp()

    # Sanitize filename for temp file
    safe_name = os.path.basename(filename) if filename else "document.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"

    pdf_path = os.path.join(out_dir, safe_name)

    try:
        with open(pdf_path, "wb") as f:
            f.write(pdf_file_bytes)

        proc = await asyncio.create_subprocess_exec(
            "libreoffice",
            "--headless",
            "--norestore",
            "--convert-to", "docx",
            "--outdir", out_dir,
            pdf_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=120
        )

        if proc.returncode != 0:
            err = stderr.decode(errors="replace") if stderr else ""
            raise RuntimeError(
                f"LibreOffice conversion failed (exit {proc.returncode}): {err[:500]}"
            )

        base = os.path.splitext(safe_name)[0]
        docx_path = os.path.join(out_dir, f"{base}.docx")

        if not os.path.exists(docx_path):
            raise RuntimeError(
                f"LibreOffice produced no output file at {docx_path}"
            )

        with open(docx_path, "rb") as f:
            return f.read()

    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
