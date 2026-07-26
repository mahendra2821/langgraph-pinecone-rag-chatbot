import os
from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME"
)


PDF_PATH = "data/Ebook-Agentic-AI.pdf"


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


TOP_K = 4