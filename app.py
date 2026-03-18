import json
import os
import streamlit as st
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="Visa Job Finder AI",
    page_icon="🌍",
)

st.title("🌍 Visa Job Finder AI")
st.write("Find AI & Data jobs with visa sponsorship using AI")

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
embeddings = OpenAIEmbeddings()

# Vector DB
vectorstore = FAISS.from_documents(docs, embeddings)

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

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
    Reason
    """

    response = llm.invoke(prompt)

    st.subheader("AI Recommendation")
    st.write(response.content)
