from typing import TypedDict

class AppState(TypedDict):
    applicant_info: str
    followup_query: str
    extracted_data: dict
    eligibility: bool
    validation_results: dict
    final_response: str