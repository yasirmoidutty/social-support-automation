# utils/eligibility.py

"""
Eligibility module for Social Support Automation.
This module decides whether an applicant qualifies for support.
"""

def check_eligibility(applicant_info):
    """
    Determines if the applicant is eligible for social support.

    Parameters:
    applicant_info (dict): Dictionary containing extracted applicant details.
        Expected keys:
            - monthly_income (int/float)
            - other_income (int/float, optional)
            - family_size (int)
            - dependents (int, optional)
            - employment_status (str, optional)
            - assets (float, optional)
            - liabilities (float, optional)

    Returns:
    bool: True if eligible, False otherwise
    """

    # Extract fields with defaults
    monthly_income = applicant_info.get("monthly_income", 0)
    other_income = applicant_info.get("other_income", 0)
    family_size = applicant_info.get("family_size", 1)  # assume at least 1
    assets = applicant_info.get("assets", 0)
    liabilities = applicant_info.get("liabilities", 0)

    total_income = monthly_income + other_income
    net_assets = assets - liabilities

    # Rule-based eligibility logic
    # You can adjust thresholds as per your requirement
    if total_income <= 10000 and family_size >= 4 and net_assets <= 50000:
        return True
    else:
        return False


# Example usage
if __name__ == "__main__":
    applicant_example = {
        "monthly_income": 7300,
        "other_income": 500,
        "family_size": 4,
        "assets": 25000,
        "liabilities": 15000
    }

    eligible = check_eligibility(applicant_example)
    print("Eligible:", eligible)
