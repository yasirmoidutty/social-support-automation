Social Support Application Automation
Overview

This project is a Python-based prototype for automating the evaluation of social support eligibility for applicants. It demonstrates an AI workflow integrating:

ML Model – structured eligibility prediction

LLM (GPT4All) – explanation and recommendations

Agentic Orchestration – modular agents simulating end-to-end workflow

Interactive Chat – live user queries

The demo uses synthetic data and mock document uploads to showcase automation, decision-making, and explainability.

Project Structure
social-support-automation/
│
├── app.py                 # Main Streamlit application
├── models/
│   ├── eligibility_model.joblib  # Trained Random Forest model
│   └── q4_0-orca-mini-3b.gguf   # Local GPT4All LLM model
├── data/
│   └── applicants.csv      # Synthetic dataset
├── utils/
│   └── llm_utils.py        # Agents for LLM explanations and data extraction
└── README.md              # Project documentation

Pain Points & How This Prototype Addresses Them
Pain Point	Prototype Solution
Manual Data Gathering	DataExtractionAgent simulates automatic extraction of key fields (age, income, employment, family size, assets) from uploaded PDFs/images, reducing manual entry errors.
Semi-Automated Data Validations	Input validation in Streamlit + ML model predictions ensures numeric ranges and eligibility rules are checked automatically.
Inconsistent Information	Future-ready ValidationAgent can compare multiple document sources. In demo, synthetic or multiple inputs can be provided to show resolution logic.
Time-Consuming Reviews	Agentic workflow runs extraction → ML prediction → LLM explanation automatically in seconds, simulating the elimination of multi-department rounds.
Subjective Decision-Making	ML model provides consistent and reproducible decisions, while LLM explains the reasoning in human-readable language, reducing human bias.
Workflow

Document Upload / Input Form: Applicant uploads supporting documents or enters details manually.

Data Extraction Agent: Extracts key features from uploaded files (simulated).

Eligibility Decision Agent (ML): Random Forest model predicts eligibility.

Recommendation Agent (LLM): Generates explanations, recommendations, and next steps.

Interactive Chat (Optional): Users can ask questions about eligibility or recommendations.

Dataset

Synthetic dataset with 200 applicants (data/applicants.csv)

Features:

age: 18–65

employment_years: 0–20

income: 2000–10000

family_size: 1–8

assets: 0–50000

Target: eligible (0 = Not Eligible, 1 = Eligible)

Eligibility rule: (income < 5000 OR assets < 10000) AND family_size > 3

Machine Learning Model

Model: Random Forest Classifier (scikit-learn)

Inputs: age, employment years, income, family size, assets

Output: eligible (binary)

Purpose: Provides consistent eligibility decision to reduce human bias.

LLM / Agents

LLM: Local GPT4All model (orca-mini-3b.gguf)

Agents:

DataExtractionAgent → extracts applicant features from uploaded documents

RecommendationAgent → generates human-readable explanation and recommendations

Role: Guides workflow, simulates agentic orchestration, explains decisions

Technology Stack

Programming Language: Python 3.11

Front-End: Streamlit (interactive chat, forms, file uploads)

ML: Scikit-learn (Random Forest)

LLM Hosting: GPT4All (local, CPU)

Agentic Orchestration: Simulated using Python agent classes

Optional / Future Enhancements:

PostgreSQL / MongoDB for storing applicant data

Redis / Qdrant for semantic search / vector storage

LangGraph / Agno / Crew.AI for production-grade agent orchestration

Langfuse for agent observability

FastAPI for serving ML/LLM endpoints

How to Run

Install dependencies:

pip install -r requirements.txt


Start Streamlit app:

streamlit run app.py


Open browser at:

http://localhost:8501


Upload documents or manually enter applicant details.

Click Check Eligibility to see ML decision and LLM explanation.

Use interactive chat to ask follow-up questions.

Future Improvements

Implement OCR + NLP for real document extraction

Cross-validate information from multiple uploaded documents

Full agentic orchestration using LangGraph / Agno

Store applicant data in databases for audit and history

Expand ML model for more sophisticated eligibility criteria