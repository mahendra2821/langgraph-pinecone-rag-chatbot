import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.config import (
    PINECONE_INDEX_NAME,
    TOP_K
)


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    return vectorstore


def retrieve_documents(question):
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(
        question,
        k=TOP_K
    )
    documents = []
    scores = []
    for doc, score in results:
        documents.append(
            doc.page_content
        )
        scores.append(
            score
        )
    return documents, scores