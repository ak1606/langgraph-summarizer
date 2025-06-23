import os
from dotenv import load_dotenv
import requests
from supabase import create_client, Client
from langchain.llms.base import LLM
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import Field
import PyPDF2

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
SUPABASE_VECTOR_TABLE = os.getenv("SUPABASE_VECTOR_TABLE", "documents")

# Gemini LLM wrapper (using REST API)
class GeminiLLM(LLM):
    api_key: str = Field(...)
    model: str = Field(default="gemini-2.0-flash")

    @property
    def _llm_type(self):
        return "gemini"

    def _call(self, prompt, **kwargs):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

# Supabase vector DB helper
class SupabaseVectorDB:
    def __init__(self, url, key, table):
        self.client: Client = create_client(url, key)
        self.table = table

    def upsert_document(self, doc_id, embedding, metadata):
        self.client.table(self.table).upsert({
            "id": doc_id,
            "embedding": embedding,
            "metadata": metadata
        }).execute()

    def query_similar(self, embedding, top_k=1):
        # This assumes you have pgvector extension and a column 'embedding' in your table
        query = f"SELECT *, (embedding <#> '{embedding}') as distance FROM {self.table} ORDER BY distance ASC LIMIT {top_k}"
        return self.client.rpc('execute_sql', {"sql": query}).execute()

# Document summarization agent
class SummarizationAgent:
    def __init__(self):
        self.llm = GeminiLLM(api_key=GEMINI_API_KEY)
        self.vector_db = SupabaseVectorDB(SUPABASE_URL, SUPABASE_API_KEY, SUPABASE_VECTOR_TABLE)

    def summarize(self, text):
        prompt = f"Summarize the following document:\n{text}"
        summary = self.llm(prompt)
        return summary

    def embed_and_store(self, doc_id, text, summary):
        # Use Gemini to get embedding (simulate with summary for now)
        dummy = [float(hash(x)%100)/100 for x in summary.split()[:1536]]
        if len(dummy) < 1536:
            dummy += [0.0] * (1536 - len(dummy))
        embedding = dummy[:1536]
        self.vector_db.upsert_document(doc_id, embedding, {"summary": summary})

    def process_document(self, doc_id, text):
        summary = self.summarize(text)
        self.embed_and_store(doc_id, text, summary)
        return summary

# For Streamlit UI
if __name__ == "__main__":
    import streamlit as st
    st.title("Document Summarization Agent (Gemini + Supabase)Upload txt, md, or pdf files to summarize")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf"])
    if uploaded_file:
        if uploaded_file.name.lower().endswith(".pdf"):
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        else:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        agent = SummarizationAgent()
        summary = agent.process_document(uploaded_file.name, text)
        st.subheader("Summary:")
        st.write(summary)
