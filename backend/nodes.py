from state import AppState
from llm import parse_applicant_info, document_validater, response_generator_ollama
import json
import ollama
import os
import joblib
import pandas as pd

model_path = os.path.join("../models", "eligibility_model.joblib")
eligibility_model = joblib.load(model_path)

def data_extractor(state: AppState) -> AppState:
    application = state.get("applicant_info", {}).get("application_form", "")
    print("Application Data\n\n--------------", application)
    if application and application.strip():
        try:
            print("Extracting relevant info")
            extracted_data = parse_applicant_info(extracted_text=application)
        except:
            print("data extraction failed")
            extracted_data = {}
    return {"extracted_data": extracted_data}

def data_validator(state: AppState) -> AppState:
    print("Validating data...")
    validation_results = document_validater(state)
    return {"validation_results": validation_results}

def eligibility_checker(state: AppState) -> AppState:
    applicant_info = state.get("extracted_data", {})

    df = pd.DataFrame([{
        "income": applicant_info.get("monthly_income", 0),
        "family_size": len(applicant_info.get("family_members", [])),
        "employment_years": applicant_info.get("employment_years", 0),
        "assets": applicant_info.get("assets", 0),
        "age": applicant_info.get("age", 0)     
    }])

    print(df)

    prediction = eligibility_model.predict(df)[0]
    eligibilty = True if prediction == 0 else False
    return {"eligibility": eligibilty}

def response_generator(state: AppState) -> AppState:
    final_response = response_generator_ollama(state)
    return {"final_response": final_response}

def orchestrator(state: AppState) -> str:
    state_json = json.dumps(state, indent=2)

    prompt = f"""
        You are an Orchestrator AI for validating a social support application.

        Workflow state:
        {state_json}

        Rules:
        1. Start with 'data_extractor' when new application_info is received.
        2. After extraction, go to 'eligibility_checker'.
        3. If eligible=True, go to 'data_validator'. then go to `response_generator`
        4. If eligible=False, go to 'response_generator'.
        5. Follow-up questions from user go to 'response_generator'.

        Output strictly in JSON:
        {{
        "next_node": "<name of next node>",
        "reason": "<short reason>"
        }}
    """

    response = ollama.chat(model="qwen2.5:7b-instruct", messages=[{"role": "user", "content": prompt}])
    
    try:
        decision = json.loads(response["content"])
    except Exception:
        decision = {"next_node": "data_extractor", "reason": "fallback due to parse error"}

    print(f"🧭 Orchestrator Decision → {decision['next_node']} ({decision['reason']})")
    return decision["next_node"]