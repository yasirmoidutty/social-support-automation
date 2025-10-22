# utils/llm_utils.py
import subprocess
import re
import json

MODEL_NAME = "qwen2.5:7b-instruct"

def query_ollama(prompt: str) -> str:
    """
    Query Ollama CLI using subprocess.
    """
    try:
        # Note: Ollama now reads input from stdin
        process = subprocess.Popen(
            ["ollama", "run", MODEL_NAME],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=prompt)
        if stderr:
            return f"Error: {stderr.strip()}"
        return stdout.strip()
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"

def extract_applicant_info(text: str) -> dict:
    """
    Extract numeric fields from the text.
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
        "assets": 8000
    }

    numbers = [clean_number(n) for n in re.findall(r'\d{1,3}(?:,\d{3})*|\d+', text)]

    if len(numbers) >= 5:
        info['age'] = numbers[0]
        info['employment_years'] = numbers[1]
        info['income'] = numbers[2]
        info['family_size'] = numbers[3]
        info['assets'] = numbers[4]

    return info

def explain_eligibility(applicant: dict) -> str:
    """
    Ask the model to explain eligibility.
    """
    prompt = f"""
The applicant has the following details:
- Age: {applicant['age']}
- Employment Years: {applicant['employment_years']}
- Monthly Income: {applicant['income']}
- Family Size: {applicant['family_size']}
- Total Assets: {applicant['assets']}
- Eligibility: {"ELIGIBLE" if applicant.get('eligible', True) else "NOT eligible"}

Explain briefly why this applicant is or is not eligible for social support.
"""
    return query_ollama(prompt)
