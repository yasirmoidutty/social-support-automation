import streamlit as st
import os
import joblib
import pandas as pd
from utils import ocr_utils, llm_utils

st.set_page_config(page_title="🛠 Social Support Application Automation", layout="wide")
st.title("🛠 Social Support Application Automation")
st.write("Upload applicant documents (PDF/Image) and check eligibility for social support.")

uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("📄 Extracting Applicant Information...")
    extracted_text = ocr_utils.extract_text_from_file(file_path)

    if extracted_text and len(extracted_text.strip()) > 0:
        st.subheader("Preview Extracted Text")
        st.text_area("Extracted Text", extracted_text, height=400)

        st.info("🧠 Parsing Applicant Information with LLM...")
        try:
            applicant_info, raw_llm = llm_utils.parse_applicant_info(extracted_text, return_raw=True)

            st.success("✅ Applicant Information Parsed Successfully")

            # Show raw LLM output
            st.subheader("Raw LLM Output")
            st.text_area("LLM Raw Response", raw_llm, height=300)

            # Show structured dictionary
            st.subheader("Structured Applicant Info")
            st.json(applicant_info)

            # --- Eligibility evaluation ---
            st.info("🏛 Evaluating Eligibility...")
            try:
                model_path = os.path.join("models", "eligibility_model.joblib")
                model = joblib.load(model_path)

                df = pd.DataFrame([{
                    "age": applicant_info.get("age", 0),
                    "income": applicant_info.get("monthly_income", 0),
                    "family_size": len(applicant_info.get("family_members", [])),
                    "liabilities": applicant_info.get("liabilities", 0),
                    "assets": applicant_info.get("assets", 0),
                    "employment_years": applicant_info.get("employment_years", 0)
                }])

                prediction = model.predict(df)[0]
                result = "✅ Eligible" if prediction == 1 else "❌ Not Eligible"
                st.subheader("Final Decision")
                st.markdown(f"### {result}")

            except Exception as e:
                st.warning(f"ML model evaluation failed: {e}")
                st.info("Using fallback rule-based eligibility check...")
                fallback_result = llm_utils.check_eligibility(applicant_info)
                st.subheader("Final Decision (Fallback)")
                st.markdown(f"### {fallback_result['reason']}")

        except Exception as e:
            st.error(f"Error during LLM parsing or evaluation: {e}")

    else:
        st.error("❌ No text could be extracted from the uploaded document.")
