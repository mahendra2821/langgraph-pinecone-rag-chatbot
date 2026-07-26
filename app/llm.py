import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # loads .env from project root

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
"""
You are an AI assistant answering questions about the Agentic AI ebook.

STRICT RULES:

1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. If the answer is not in the context, say:
"I could not find this information in the provided Agentic AI ebook."

Context:

{context}


Question:

{question}


Answer:
"""
)


def generate_answer(question, documents):

    context = "\n\n".join(
        documents
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return response.content