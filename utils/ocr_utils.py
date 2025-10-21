import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io

def extract_text_from_file(uploaded_file):
    """
    Extract text from uploaded PDF or image file.
    """
    filename = uploaded_file.name
    if filename.endswith(".pdf"):
        # Convert PDF pages to images
        images = convert_from_path(uploaded_file)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
        return text
    else:  # PNG or JPG
        img = Image.open(uploaded_file)
        return pytesseract.image_to_string(img)
