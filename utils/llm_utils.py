# utils/llm_utils.py
import json
import re
from typing import Dict, Any
import ollama

MODEL_NAME = "qwen2.5:7b-instruct"  # make sure you pulled this locally

# -------------------------------
# 1️⃣ Parse Applicant Information with LLM
# -------------------------------
def parse_applicant_info(extracted_text: str) -> Dict[str, Any]:
    """
    Uses local Ollama LLM to extract structured applicant information
    from OCR text. Falls back to regex if LLM fails.
    """
    if not extracted_text or len(extracted_text.strip()) == 0:
        raise ValueError("No text provided for parsing.")

    # Limit input length to 2000 chars to avoid model truncation
    short_text = extracted_text[:2000]

    prompt = f"""
You are an expert document parser. Extract structured applicant information
from the following social support application text.

OCR Text:
\"\"\"{short_text}\"\"\"

Return a valid JSON with these fields:
{{
    "name": "",
    "age": "",
    "gender": "",
    "marital_status": "",
    "employment_status": "",
    "monthly_income": "",
    "family_members": "",
    "address": "",
    "disability_status": "",
    "other_support_received": ""
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

        # Debug: see what LLM actually returned
        print("[DEBUG] LLM raw response:", response)

        text_output = response.get("message", {}).get("content", "").strip()

        # Try parsing JSON from LLM output
        match = re.search(r"\{.*\}", text_output, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return parsed
        else:
            raise ValueError("LLM did not return valid JSON")

    except Exception as e:
        print(f"[WARN] LLM parsing failed: {e}")
        # Fallback to regex parser
        return fallback_parse(extracted_text)

# -------------------------------
# 2️⃣ Fallback Parser (regex)
# -------------------------------
def fallback_parse(text: str) -> Dict[str, Any]:
    """
    Basic regex fallback parser when LLM fails.
    Improved to handle long names and realistic fields.
    """
    def extract(pattern):
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else ""

    return {
        "name": extract(r"Full Name[:\-]?\s*(.+)"),
        "age": extract(r"Age[:\-]?\s*(\d+)"),
        "gender": extract(r"Gender[:\-]?\s*(Male|Female|Other)"),
        "marital_status": extract(r"Marital Status[:\-]?\s*(.+)"),
        "employment_status": extract(r"Employment Status[:\-]?\s*(.+)"),
        "monthly_income": extract(r"Monthly Income.*[:\-]?\s*([\d,]+)"),
        "family_members": extract(r"Family Size.*[:\-]?\s*(\d+)"),
        "address": extract(r"Address[:\-]?\s*(.+)"),
        "disability_status": extract(r"Disability[:\-]?\s*(.+)"),
        "other_support_received": extract(r"Support[:\-]?\s*(.+)"),
    }

# -------------------------------
# 3️⃣ Eligibility Check (optional)
# -------------------------------
def check_eligibility(applicant_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple rule-based eligibility check.
    """
    try:
        income = int(str(applicant_data.get("monthly_income", "0")).replace(",", ""))
        family_members = int(applicant_data.get("family_members", 1))
    except Exception:
        income, family_members = 0, 1

    eligible = income < 25000 and family_members >= 3
    reason = (
        "✅ Eligible for support"
        if eligible
        else "❌ Not eligible — income exceeds limit or family size too small."
    )

    return {"eligible": eligible, "reason": reason}

