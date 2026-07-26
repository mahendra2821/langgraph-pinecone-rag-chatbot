from app.graph import rag_graph


response = rag_graph.invoke(
    {
        "question": "What is Agentic AI?"
    }
)


print("\nANSWER:")
print(response["answer"])


print("\nSOURCES:")

for doc in response["documents"]:
    print("----------------")
    print(doc[:300])


print("\nCONFIDENCE:")

if response["scores"]:
    print(sum(response["scores"]) / len(response["scores"]))
else:
    print("No documents retrieved — check Pinecone index contents.")