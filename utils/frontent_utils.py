import os
from utils.ocr_utils import extract_text_from_file

def process_uploaded_files(uploaded_files):
    info = {}
    for file in uploaded_files:
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        text = extract_text_from_file(file_path)

        if "application" in file.name.lower():
            info["application_form"] = text
        elif "passport" in file.name.lower():
            info["passport"] = text
        elif "salary" in file.name.lower():
            info["salary_slip"] = text 
        elif "bank" in file.name.lower():
            info["bank_statement"] = text

    return info
