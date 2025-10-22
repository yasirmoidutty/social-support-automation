import streamlit as st
import re
import fitz  # PyMuPDF
from PIL import Image
import io

# ----------------------------
# Utility functions
# ----------------------------

def extract_text_from_pdf(file):
    text = ""
    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def extract_text_from_image(file):
    try:
        image = Image.open(file)
        # If you have OCR setup, you can integrate pytesseract here
        # For now, just note that we need OCR
        return "[OCR required for image extraction]"
    except Exception as e:
        st.error(f"Error reading image: {e}")
        return ""

def extract_applicant_info(text):
    """
    Extract numeric fields from text using regex.
    Default values are provided for missing info.
    """
    def clean_number(n):
        try:
            return int(n.replace(',', '').strip())
        except:
            return 0

    info = {
        "age": 30,
        "employment_years": 5,
        "income": 4000,
        "family_size": 4,
        "assets": 8000,
        "eligible": True
    }

    # Extract numbers including commas
    numbers = [clean_number(n) for n in re.findall(r'\d{1,3}(?:,\d{3})*|\d+', text)]

    if len(numbers) >= 5:
        info['age'] = numbers[0]
        info['employment_years'] = numbers[1]
        info['income'] = numbers[2]
        info['family_size'] = numbers[3]
        info['assets'] = numbers[4]

    # Simple eligibility logic
    if info['income'] > 10000:
        info['eligible'] = False

    return info

def explain_eligibility(applicant):
    """
    Dummy LLM explanation.
    Replace this with your Ollama / other LLM integration.
    """
    if applicant['eligible']:
        return "Applicant qualifies for social support due to moderate income and family size."
    else:
        return "Applicant does not qualify due to high income."

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🛠 Social Support Application Automation")
st.write("Upload applicant documents (PDF/Image) and check eligibility for social support.")

uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.info("Extracting Applicant Information...")
    if uploaded_file.type == "application/pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    else:
        extracted_text = extract_text_from_image(uploaded_file)

    # Show extracted text
    st.text_area("Extracted Text", extracted_text, height=300, label_visibility="hidden")

    # Extract numeric info and check eligibility
    applicant_info = extract_applicant_info(extracted_text)

    # Show eligibility
    if applicant_info['eligible']:
        st.success("✅ Applicant is ELIGIBLE for social support.")
    else:
        st.error("❌ Applicant is NOT eligible for social support.")

    # LLM explanation
    explanation = explain_eligibility(applicant_info)
    st.text_area("LLM Explanation", explanation, height=150, label_visibility="hidden")
