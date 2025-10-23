import json
import re
from typing import Dict, Any
import ollama
from datetime import date

MODEL_NAME = "qwen2.5:7b-instruct"

def parse_applicant_info(extracted_text: str, return_raw: bool = False) -> Dict[str, Any]:
    if not extracted_text.strip():
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
            stream=False
        )

        text_output = response["message"]["content"].strip()
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

def fallback_parse(text: str) -> Dict[str, Any]:
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
        "family_members": int(extract(r"Family Members[:\-]?\s*(\d+)") or 1),
        "address": extract(r"Address[:\-]?\s*(.*)"),
        "disability_status": "",
        "other_support_received": "",
        "assets": int(extract(r"Total Assets.*?[:\-]?\s*([\d,]+)") or 0),
        "liabilities": int(extract(r"Total Liabilities.*?[:\-]?\s*([\d,]+)") or 0)
    }

def check_eligibility(applicant_data: Dict[str, Any]) -> Dict[str, Any]:
    income = applicant_data.get("monthly_income", 0)
    family_members = applicant_data.get("family_members", 1)
    assets = applicant_data.get("assets", 0)
    liabilities = applicant_data.get("liabilities", 0)

    eligible = income < 25000 and family_members >= 3 and (assets - liabilities) <= 50000
    reason = "✅ Eligible for support" if eligible else "❌ Not eligible — check income/family/assets."

    return {"eligible": eligible, "reason": reason}

def document_validater(state: Dict[str, Any]) -> Dict[str, Any]:
    application_data = state.get("extracted_data", {})
    uploaded_docs = state.get("applicant_info", {})

    age = application_data.get("age", 0)
    income = application_data.get("monthly_income", 0)

    salary_slip_text = uploaded_docs.get("salary", "")
    passport_text = uploaded_docs.get("passport", "")
    today = date.today()

    prompt = f"""
You are a Data Validator AI.

Validate the social support application.

### Inputs:
Application data:
Age: {age}
Income: {income}

Uploaded documents:
Salary Slip: {salary_slip_text}
Passport: {passport_text}

Validation Rules:
1. Age: Calculate age from passport DOB and today's date {today}. Compare with application age.
2. Income: Compare application income with salary slip income.

Output strictly as JSON:
{{
"age_validation": "success" | "failed",
"income_validation": "success" | "failed",
"overall_status": "success" | "failed",
"reason": "<reason if failed>"
}}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        validation_result = json.loads(response.get("content", "{}"))
    except Exception as e:
        print(f"[WARN] Validation parsing failed: {e}")
        validation_result = {
            "age_validation": "failed",
            "income_validation": "failed",
            "overall_status": "failed",
            "reason": "Failed to parse model output"
        }

    return validation_result

def response_generator_ollama(state: Dict[str, Any]) -> Dict[str, Any]:
    eligibility = state.get("eligibility", False)
    data_validation = state.get("validation_results", {})

    eligibility_json = json.dumps(eligibility)
    validation_json = json.dumps(data_validation)

    prompt = f"""
You are a Response Generator AI for a social support application.

### Inputs:
eligibility = {eligibility_json}
data_validation = {validation_json}

### Rules:
- Applicant is eligible if:
    1. eligibility is True AND
    2. data_validation["overall_status"] is "success"
- Otherwise, not eligible.
- Provide a clear reason explaining the decision.

### Output strictly as JSON:
{{
"final_status": "eligible" | "not eligible",
"reason": "<short reason explaining eligibility or failure>"
}}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        final_response = json.loads(response.get("content", "{}"))
    except Exception as e:
        print(f"[WARN] Response parsing failed: {e}")
        final_response = {"final_status": "not eligible", "reason": "Failed to parse model output"}

    return final_response
