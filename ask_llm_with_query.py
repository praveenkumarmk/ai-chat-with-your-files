import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = ChatOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    model="openai/gpt-oss-120b",
    temperature=0.3,
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF assistant. Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I couldn't find that in the document."
After your answer, list the page numbers you used as: Sources: page X, page Y.

Context:
{context}

Question: {question}
""")

def ask(db, question):
    chunks  = db.similarity_search(question, k=4)
    context = "\n\n".join(
        f"[page {c.metadata['page']+1}] {c.page_content}" for c in chunks)
    chain = prompt | model
    return chain.invoke({"context": context, "question": question}).content