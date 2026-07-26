from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.graph import rag_graph

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    version="1.0"
)

# Allow the frontend to call this API from the browser.
# Tighten allow_origins to your actual frontend URL before deploying to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    contexts: list[str]
    confidence: float


@app.get("/")
def home():
    return {
        "message": "Agentic AI RAG chatbot running"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = rag_graph.invoke(
            {
                "question": request.question
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    scores = result.get("scores", [])
    confidence = round(sum(scores) / len(scores), 3) if scores else 0.0

    return {
        "answer": result["answer"],
        "contexts": result["documents"],
        "confidence": confidence
    }