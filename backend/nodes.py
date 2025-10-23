from state import AppState
from llm import parse_applicant_info, document_validater, response_generator_ollama, check_eligibility
import json
import ollama
import os
import joblib
import pandas as pd
import re

# Load eligibility model once
model_path = os.path.join("../models", "eligibility_model.joblib")
eligibility_model = joblib.load(model_path)



class DataExtractor:
    def __call__(self, state: AppState) -> dict:
        application = state.get("applicant_info", {}).get("application_form", "")
        if not application.strip():
            uploaded = state.get("applicant_info", {})
            application = " ".join([
                uploaded.get("bank_statement", ""),
                uploaded.get("salary", ""),
                uploaded.get("passport", "")
            ])

        print("Application Data\n\n--------------", application)

        extracted_data = {}
        if application.strip():
            try:
                print("Extracting relevant info...")
                extracted_data = parse_applicant_info(extracted_text=application)
            except Exception as e:
                print("Data extraction failed:", e)

        print("Extracted Data:", extracted_data)
        return {"extracted_data": extracted_data}


class DataValidator:
    def __call__(self, state: AppState) -> dict:
        print("Validating data...")
        validation_results = document_validater(state)
        return {"validation_results": validation_results}


class EligibilityChecker:
    def __call__(self, state: AppState) -> dict:
        applicant_info = state.get("extracted_data", {})

        # Convert monthly_income to number
        raw_income = applicant_info.get("monthly_income", 0)
        if isinstance(raw_income, str):
            numbers = re.findall(r"[\d,]+", raw_income)
            income = int(numbers[0].replace(",", "")) if numbers else 0
        else:
            income = raw_income

        # Family members
        family_members = applicant_info.get("family_members", 1)
        try:
            family_size = int(family_members)
        except Exception:
            family_size = 1

        # Assets
        raw_assets = applicant_info.get("assets", 0)
        if isinstance(raw_assets, str):
            numbers = re.findall(r"[\d,]+", raw_assets)
            assets = int(numbers[0].replace(",", "")) if numbers else 0
        else:
            assets = raw_assets

        # Liabilities
        raw_liabilities = applicant_info.get("liabilities", 0)
        if isinstance(raw_liabilities, str):
            numbers = re.findall(r"[\d,]+", raw_liabilities)
            liabilities = int(numbers[0].replace(",", "")) if numbers else 0
        else:
            liabilities = raw_liabilities

        df = pd.DataFrame([{
            "income": income,
            "family_size": family_size,
            "employment_years": applicant_info.get("employment_years", 0),
            "assets": assets,
            "age": applicant_info.get("age", 0)
        }])

        print("Eligibility Input DataFrame:\n", df)

        try:
            prediction = eligibility_model.predict(df)[0]
            eligibility = True if prediction == 0 else False
        except Exception as e:
            print("Eligibility model failed:", e)
            eligibility_result = check_eligibility({
                "monthly_income": income,
                "family_members": family_size,
                "assets": assets,
                "liabilities": liabilities
            })
            eligibility = eligibility_result.get("eligible", False)

        return {"eligibility": eligibility}


class ResponseGenerator:
    def __call__(self, state: AppState) -> dict:
        final_response = response_generator_ollama(state)
        return {"final_response": final_response}


class Orchestrator:
    def __call__(self, state: AppState) -> dict:
        state_json = json.dumps(state, indent=2)
        prompt = f"""
        You are an Orchestrator AI for validating a social support application.

        Workflow state:
        {state_json}

        Rules:
        1. Start with 'data_extractor' when new application_info is received.
        2. After extraction, go to 'eligibility_checker'.
        3. If eligible=True, go to 'data_validator'. then go to `response_generator`
        4. If eligible=False, or `validation_results` available go to 'response_generator'.
        5. Follow-up questions from user go to 'response_generator'.

        Output strictly in JSON:
        {{
        "next_node": "<name of next node>",
        "reason": "<short reason>"
        }}
        """
        try:
            response = ollama.chat(model="qwen2.5:7b-instruct", messages=[{"role": "user", "content": prompt}])
            decision = json.loads(response.get("content", "{}"))
        except Exception as e:
            print("Orchestrator failed:", e)
            decision = {"next_node": "data_extractor", "reason": "fallback due to parse error"}

        print(f"🧭 Orchestrator Decision → {decision.get('next_node')} ({decision.get('reason')})")
        return {
            **state,
            "next": decision.get("next_node", "data_extractor")
        }
