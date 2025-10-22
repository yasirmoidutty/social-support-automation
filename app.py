import streamlit as st
import os
import joblib
from utils import ocr_utils, llm_utils, eligibility
import pandas as pd

st.set_page_config(page_title="🛠 Social Support Application Automation", layout="wide")

st.title("🛠 Social Support Application Automation")
st.write("Upload applicant documents (PDF/Image) and check eligibility for social support.")

uploaded_file = st.file_uploader(
    "Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    # Save temporarily
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
            applicant_info = llm_utils.parse_applicant_info(extracted_text)
            st.success("✅ Applicant Information Parsed Successfully")

            # Display raw JSON
            st.subheader("Raw Parsed Output")
            st.json(applicant_info)

            # Fields filled by fallback parser
            fallback_fields = []
            for key, value in applicant_info.items():
                if value in ("", [], None):
                    fallback_fields.append(key)
            st.write("Fields filled by fallback parser:", fallback_fields)

            # Structured info
            st.subheader("Structured Applicant Info")
            st.json(applicant_info)

            # --- Eligibility evaluation ---
            st.info("🏛 Evaluating Eligibility...")

            try:
                # Load trained ML model
                model_path = os.path.join("models", "eligibility_model.joblib")
                model = joblib.load(model_path)

                # Prepare DataFrame for ML prediction
                df = pd.DataFrame([{
                    "age": int(applicant_info.get("age", 0)),
                    "income": float(str(applicant_info.get("monthly_income", 0)).replace(",", "")),
                    "family_size": len(applicant_info.get("family_members", [])),
                    "liabilities": float(applicant_info.get("liabilities", 0)),
                    "assets": float(applicant_info.get("assets", 0))
                }])

                # Predict eligibility using ML model
                prediction = model.predict(df)[0]
                result = "✅ Eligible" if prediction == 1 else "❌ Not Eligible"
                st.subheader("Final Decision (ML Model)")
                st.markdown(f"### {result}")

            except Exception as ml_error:
                st.warning(f"ML model evaluation failed: {ml_error}")
                st.info("Using fallback rule-based eligibility check...")

                # Fallback rule-based eligibility
                fallback_result = eligibility.check_eligibility({
                    "monthly_income": float(str(applicant_info.get("monthly_income", 0)).replace(",", "")),
                    "other_income": float(str(applicant_info.get("other_support_received", 0)).replace(",", "")) or 0,
                    "family_size": len(applicant_info.get("family_members", [])),
                    "assets": float(applicant_info.get("assets", 0)),
                    "liabilities": float(applicant_info.get("liabilities", 0))
                })
                result_text = "✅ Eligible" if fallback_result else "❌ Not Eligible"
                st.subheader("Final Decision (Fallback Rule)")
                st.markdown(f"### {result_text}")

        except Exception as e:
            st.error(f"Error during LLM parsing or evaluation: {e}")

    else:
        st.error("❌ No text could be extracted from the uploaded document.")
