# utils/ocr_utils.py
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

# 🧠 Function to extract text from PDF or image file
def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from a PDF or image file using PyMuPDF and Tesseract OCR.
    Supports both scanned PDFs and image uploads.
    """
    try:
        text = ""
        file_ext = os.path.splitext(file_path)[1].lower()

        # ---- Case 1: PDF ----
        if file_ext == ".pdf":
            with fitz.open(file_path) as pdf:
                for page_index in range(len(pdf)):
                    page = pdf.load_page(page_index)

                    # Try to extract text directly
                    page_text = page.get_text("text")

                    # If no text (i.e., scanned), use OCR on image
                    if not page_text.strip():
                        pix = page.get_pixmap()
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        page_text = pytesseract.image_to_string(img)

                    text += page_text + "\n"

        # ---- Case 2: Image ----
        elif file_ext in [".png", ".jpg", ".jpeg"]:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)

        else:
            raise ValueError("Unsupported file type. Please upload PDF or image.")

        if not text.strip():
            raise ValueError("No text extracted — check file quality or format.")

        return text.strip()

    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""
