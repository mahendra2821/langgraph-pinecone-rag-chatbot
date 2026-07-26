import os

# Suppress noisy TensorFlow logs (optional, cosmetic)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone

from app.config import *
from app.utils import load_pdf


def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_text(text)

    return chunks


def upload_to_pinecone(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    os.environ.setdefault("PINECONE_API_KEY", PINECONE_API_KEY)

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        raise ValueError(
            f"Index '{PINECONE_INDEX_NAME}' does not exist in Pinecone. "
            f"Create it first with dimension=384 (required for all-MiniLM-L6-v2), "
            f"either in the Pinecone console or via pc.create_index(...)."
        )

    metadatas = [
        {"source": PDF_PATH, "chunk_index": i}
        for i in range(len(chunks))
    ]

    PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        metadatas=metadatas
    )

    print("Successfully uploaded to Pinecone")


if __name__ == "__main__":
    print("Loading PDF...")

    text = load_pdf(PDF_PATH)

    print("Creating chunks...")

    chunks = create_chunks(text)

    print(f"Created {len(chunks)} chunks")

    upload_to_pinecone(chunks)