import json
import re
from typing import Dict, Any, Tuple
import ollama  # ✅ Using local Ollama

MODEL_NAME = "qwen2.5:7b-instruct"  # make sure it's pulled: ollama pull qwen2.5:7b-instruct


# -------------------------------
# 1️⃣ Parse Applicant Information
# -------------------------------
def parse_applicant_info(extracted_text: str, return_raw: bool = False) -> Any:
    """
    Uses the local Ollama LLM to extract structured applicant information
    (name, age, income, etc.) from OCR text.
    If return_raw=True, returns a tuple (parsed_dict, raw_text).
    """
    if not extracted_text or len(extracted_text.strip()) == 0:
        raise ValueError("No text provided for parsing.")

    prompt = f"""
You are an expert document parser. Extract structured applicant information
from the following social support application text.

OCR Text:
\"\"\"{extracted_text}\"\"\"

Return a valid JSON with these fields:
{{
    "name": "",
    "age": "",
    "gender": "",
    "marital_status": "",
    "employment_status": "",
    "employment_years": "",
    "monthly_income": "",
    "family_members": "",
    "address": "",
    "disability_status": "",
    "other_support_received": "",
    "assets": "",
    "liabilities": ""
}}

Only return JSON. Do not add any explanations.
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts form data."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )

        text_output = response["message"]["content"].strip()

        # Clean and parse JSON safely
        match = re.search(r"\{.*\}", text_output, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            raise ValueError("LLM did not return valid JSON")

        if return_raw:
            return parsed, text_output
        return parsed

    except Exception as e:
        print(f"[WARN] LLM parsing failed: {e}")
        parsed = fallback_parse(extracted_text)
        if return_raw:
            return parsed, ""
        return parsed


# -------------------------------
# 2️⃣ Simple Fallback Parsing
# -------------------------------
def fallback_parse(text: str) -> Dict[str, Any]:
    """Regex fallback parsing when LLM fails."""
    def extract(pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    return {
        "name": extract(r"Name[:\-]?\s*([A-Za-z ]+)"),
        "age": int(extract(r"Age[:\-]?\s*(\d+)") or 0),
        "gender": extract(r"Gender[:\-]?\s*(Male|Female|Other)"),
        "marital_status": extract(r"Marital Status[:\-]?\s*([A-Za-z ]+)"),
        "employment_status": extract(r"Employment Status[:\-]?\s*([A-Za-z ]+)"),
        "employment_years": int(extract(r"Years of Employment[:\-]?\s*(\d+)") or 0),
        "monthly_income": int(extract(r"Monthly Income.*?[:\-]?\s*([\d,]+)") or 0),
        "family_members": [],  # fallback empty list
        "address": extract(r"Address[:\-]?\s*(.*)"),
        "disability_status": "",
        "other_support_received": "",
        "assets": int(extract(r"Total Assets.*?[:\-]?\s*([\d,]+)") or 0),
        "liabilities": int(extract(r"Total Liabilities.*?[:\-]?\s*([\d,]+)") or 0)
    }


# -------------------------------
# 3️⃣ Eligibility Check (optional)
# -------------------------------
def check_eligibility(applicant_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simple rule-based eligibility."""
    income = applicant_data.get("monthly_income", 0)
    family_members = len(applicant_data.get("family_members", [])) or 1
    assets = applicant_data.get("assets", 0)
    liabilities = applicant_data.get("liabilities", 0)

    eligible = income < 25000 and family_members >= 3 and (assets - liabilities) <= 50000
    reason = (
        "✅ Eligible for support"
        if eligible
        else "❌ Not eligible — check income/family/assets."
    )

    return {"eligible": eligible, "reason": reason}
