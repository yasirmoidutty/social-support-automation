import os
import warnings
import logging
from gpt4all import GPT4All
import re

# ----------------------------
# Suppress warnings
# ----------------------------
warnings.filterwarnings("ignore")
logging.getLogger("gpt4all").setLevel(logging.ERROR)
os.environ["GPT4ALL_LOG_LEVEL"] = "ERROR"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

# ----------------------------
# Absolute path to local LLM model
# ----------------------------
model_path = os.path.abspath("models/q4_0-orca-mini-3b.gguf")

# ----------------------------
# Load local GPT4All model (offline)
# ----------------------------
try:
    model = GPT4All(model_path, allow_download=False, device="cpu")
    print("✅ LLM model loaded locally")
except Exception as e:
    print(f"❌ Failed to load local LLM model: {e}")
    raise e

# ----------------------------
# Function to generate LLM explanation
# ----------------------------
def explain_eligibility(applicant):
    prompt = f"""
    The applicant has the following details:
    - Age: {applicant['age']}
    - Employment Years: {applicant['employment_years']}
    - Monthly Income: {applicant['income']}
    - Family Size: {applicant['family_size']}
    - Total Assets: {applicant['assets']}
    - Eligibility: {"ELIGIBLE" if applicant['eligible'] else "NOT eligible"}

    Please explain briefly why this applicant is or is not eligible for social support.
    """
    response = model.generate(prompt, max_tokens=120)
    return response.strip()

# ----------------------------
# Function to extract structured applicant info from uploaded documents
# ----------------------------
def extract_applicant_info(text):
    """
    For demo purposes: extract numeric fields from text using regex.
    Can be replaced with LLM extraction for real-world scenarios.
    """
    # Default values
    info = {
        "age": 30,
        "employment_years": 5,
        "income": 4000,
        "family_size": 4,
        "assets": 8000
    }

    # Extract numbers from text
    numbers = list(map(int, re.findall(r'\d+', text)))
    if len(numbers) >= 5:
        info['age'] = numbers[0]
        info['employment_years'] = numbers[1]
        info['income'] = numbers[2]
        info['family_size'] = numbers[3]
        info['assets'] = numbers[4]

    return info
