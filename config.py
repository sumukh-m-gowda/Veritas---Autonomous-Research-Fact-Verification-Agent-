from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

REQUIRED_ENV_VARS = ["GEMINI_API_KEY"]

for var in REQUIRED_ENV_VARS:
    if var not in os.environ:
        raise RuntimeError(
            f"Missing required environment variable '{var}'. "
            f"Copy .env.example to .env and fill it in."
        )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GEMINI_API_KEY"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "veritas.db")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)