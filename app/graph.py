from typing import TypedDict, List

from langgraph.graph import (
    StateGraph,
    END
)


from app.rag import retrieve_documents
from app.llm import generate_answer



class RAGState(TypedDict):

    question: str

    documents: List[str]

    scores: List[float]

    answer: str



# NODE 1
def retrieve_node(state):


    docs, scores = retrieve_documents(

        state["question"]

    )


    return {

        "documents": docs,

        "scores": scores

    }




# NODE 2
def generate_node(state):


    answer = generate_answer(

        state["question"],

        state["documents"]

    )


    return {

        "answer": answer

    }




workflow = StateGraph(
    RAGState
)



workflow.add_node(
    "retrieve",
    retrieve_node
)


workflow.add_node(
    "generate",
    generate_node
)



workflow.set_entry_point(
    "retrieve"
)



workflow.add_edge(
    "retrieve",
    "generate"
)



workflow.add_edge(
    "generate",
    END
)



rag_graph = workflow.compile()