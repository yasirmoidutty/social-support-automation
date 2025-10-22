from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import uvicorn
from graph import evaluater

app = FastAPI()


class InputData(BaseModel):
    data: Dict[str, str]
    followup_query: str

@app.post("/check_eligibility")
def process_data(input_data: InputData):
    applicant_info = input_data.data

    result = evaluater.invoke({"applicant_info": applicant_info})
    print(result)


    response = result.get("final_response")
    return response

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
