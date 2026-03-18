import json
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# Load jobs
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

# Create embeddings
embeddings = OllamaEmbeddings(model="tinyllama")


# Vector database
vectorstore = FAISS.from_documents(docs, embeddings)

# LLM
llm = Ollama(model="tinyllama")
while True:

    query = input("\nSearch jobs: ")

    if query == "exit":
        break

    results = vectorstore.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in results])

prompt = f"""
You are a job recommendation assistant.

Based on the following job listings, recommend the best job.

Provide:
- Job Title
- Company
- Location
- Visa Sponsorship
- Reason for recommendation

Job Listings:
{context}

User Query: {query}
"""

response = llm.invoke(prompt)

print("\nAI Recommendation:\n")
print(response)