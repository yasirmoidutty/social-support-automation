import streamlit as st
import os
import joblib
import pandas as pd
import requests
from utils import frontent_utils

st.set_page_config(page_title="🛠 Social Support Application Automation", layout="wide")
st.title("🛠 Social Support Application Automation")
st.write("Upload applicant documents (PDF/Image) and check eligibility for social support.")

uploaded_files = st.file_uploader("Upload Application form, Passport, Salary slip and Bank Statement", 
                                 type=["pdf", "png", "jpg", "jpeg"],
                                 accept_multiple_files=True)

if uploaded_files:
    st.info("📄 Extracting Applicant Information...")
    document_info = frontent_utils.process_uploaded_files(uploaded_files)
    print(document_info)
    payload = {"data": document_info, "followup_query": ""}

    try:
        st.info("🏛 Evaluating Eligibility...")
        response = requests.post("http://127.0.0.1:8000/check_eligibility", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success("Response from backend:")
            st.json(result)
        else:
            st.error(f"Error: {response.status_code}")

    except Exception as e:
        st.error(f"Request failed: {e}")