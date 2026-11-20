"""LangGraph workflow for the local AI research assistant."""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from src.retriever import DocumentRetriever


class ResearchState(TypedDict, total=False):
    question: str
    context: list
    answer: str
    sources: list


def create_workflow(chunks, llm=None):
    """Create a two-step retrieval and answer generation workflow."""

    retriever = DocumentRetriever(chunks)

    if llm is None:
        llm = ChatOllama(
            model="qwen2.5:3b",
            temperature=0
        )

    # Node 1: Retrieve relevant research documents.
    def retrieve_node(state: ResearchState):

        question = state["question"]

        results = retriever.search(
            query=question,
            top_k=3
        )

        sources = list(dict.fromkeys(
            item["source"] for item in results
        ))

        return {
            "context": results,
            "sources": sources
        }

    # Node 2: Generate an answer using the retrieved context.
    def generate_node(state: ResearchState):

        context = "\n\n".join(
            item["content"] for item in state.get("context", [])
        )

        if not context:
            return {
                "answer": "I could not find relevant information in the research documents."
            }

        prompt = f"""
You are a helpful AI research assistant.

Answer the user's question using the supplied research context.

Do not invent facts or unsupported information.
If the context does not contain the answer, say so.
Keep the answer clear and easy to understand.

Research Context:
{context}

User Question:
{state['question']}

Answer:
"""

        response = llm.invoke(prompt)

        return {
            "answer": response.content
        }

    graph = StateGraph(ResearchState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()