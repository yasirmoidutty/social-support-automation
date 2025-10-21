import os
import warnings
import logging
import streamlit as st
import pandas as pd
import pdfplumber
from PIL import Image
import pytesseract
from joblib import load
from utils import llm_utils

# ----------------------------
# Suppress warnings and disable GPU
# ----------------------------
warnings.filterwarnings("ignore")
logging.getLogger("gpt4all").setLevel(logging.ERROR)
os.environ["GPT4ALL_LOG_LEVEL"] = "ERROR"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU

# ----------------------------
# Load eligibility ML model
# ----------------------------
eligibility_model_path = os.path.abspath("models/eligibility_model.joblib")
model_ml = load(eligibility_model_path)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🛠 Social Support Application Automation")
st.write("Upload applicant documents (PDF/Image) and interact with AI for eligibility evaluation.")

uploaded_file = st.file_uploader(
    "Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"]
)

def extract_text(file):
    """Extract text from uploaded PDF or image"""
    text = ""
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    else:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
    return text

def extract_applicant_info(text):
    """Extract applicant info from text using simple parsing or regex"""
    info = {}
    try:
        import re
        info['age'] = int(re.search(r"Age:\s*(\d+)", text).group(1))
        info['employment_years'] = int(re.search(r"Employment Years:\s*(\d+)", text).group(1))
        info['income'] = int(re.search(r"Monthly Income:\s*(\d+)", text).group(1))
        info['family_size'] = int(re.search(r"Family Size:\s*(\d+)", text).group(1))
        info['assets'] = int(re.search(r"Total Assets:\s*(\d+)", text).group(1))
    except Exception as e:
        st.error(f"Failed to extract applicant info: {e}")
    return info

if uploaded_file:
    st.write("Extracting Applicant Information...")
    extracted_text = extract_text(uploaded_file)
    st.text_area("Preview Extracted Text", extracted_text, height=200)

    applicant_info = extract_applicant_info(extracted_text)
    st.json(applicant_info)

    if applicant_info:
        # Prepare input DataFrame for ML model
        input_df = pd.DataFrame([applicant_info])
        # Ensure feature order matches training
        input_df = input_df[['income', 'family_size', 'employment_years', 'assets', 'age']]

        # Predict eligibility
        prediction = model_ml.predict(input_df)[0]
        eligible = bool(prediction)

        if eligible:
            st.success("✅ Applicant is ELIGIBLE for social support.")
        else:
            st.error("❌ Applicant is NOT eligible for social support.")

        # Generate LLM explanation
        with st.spinner("Generating explanation with LLM..."):
            explanation = llm_utils.explain_eligibility({**applicant_info, "eligible": eligible})
        st.markdown("### 🧠 LLM Explanation")
        st.write(explanation)
