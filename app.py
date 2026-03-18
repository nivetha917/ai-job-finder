import json
import streamlit as st
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Load job dataset
with open("data_jobs.json") as f:
    jobs = json.load(f)

docs = []

for job in jobs:
    content = f"""
    Title: {job['title']}
    Company: {job['company']}
    Location: {job['location']}
    Visa Sponsorship: {job['visa']}
    Skills: {job['skills']}
    Description: {job['description']}
    """
    docs.append(Document(page_content=content))

# Embeddings
embeddings = OllamaEmbeddings(model="tinyllama")

# Vector DB
vectorstore = FAISS.from_documents(docs, embeddings)

# LLM
llm = OllamaLLM(model="tinyllama")

# Streamlit UI
st.title("🤖 AI Job Finder")
st.write("Search for jobs using AI")

query = st.text_input("Search jobs")

if st.button("Search"):

    results = vectorstore.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in results])

    prompt = f"""
    Recommend the best jobs based on this query.

    Query: {query}

    Jobs:
    {context}

    Return:
    Job Title
    Company
    Location
    Visa Sponsorship
    """

    response = llm.invoke(prompt)

    st.subheader("AI Recommendation")
    st.write(response)