import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = Path(
    os.getenv("DOCUMENTS_DIRECTORY", str(BASE_DIR / "data" / "documents"))
)

CHROMA_DIR = Path(
    os.getenv("CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "chroma_db"))
)

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME", "week8_production_rag"
)

OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
)

LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "nomic-embed-text"
)

TOP_K = int(os.getenv("TOP_K", "3"))


def get_embeddings():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL
    )


def get_vector_store():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR)
    )


def load_documents() -> list[Document]:
    documents = []

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    for path in DOCUMENTS_DIR.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix == ".txt":
            loader = TextLoader(
                str(path),
                encoding="utf-8"
            )

        elif suffix == ".pdf":
            loader = PyPDFLoader(str(path))

        else:
            continue

        loaded = loader.load()

        for document in loaded:
            document.metadata["source"] = path.name

        documents.extend(loaded)

    return documents


def ingest_documents() -> dict:
    documents = load_documents()

    if not documents:
        raise ValueError(
            "No PDF or TXT documents found in the documents directory."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    vector_store = get_vector_store()

    vector_store.add_documents(chunks)

    return {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "collection": COLLECTION_NAME
    }


class RAGState(TypedDict):
    question: str
    context: list[Document]
    answer: str


def retrieve_node(state: RAGState):
    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        state["question"],
        k=TOP_K
    )

    return {"context": documents}


def generate_node(state: RAGState):
    context = "\n\n".join(
        document.page_content
        for document in state["context"]
    )

    if not context.strip():
        return {
            "answer": "I could not find relevant information in the knowledge base."
        }

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful document question-answering assistant. "
            "Answer using only the supplied context. "
            "If the context does not contain the answer, clearly say "
            "that the information is not available in the documents. "
            "Do not invent facts."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ])

    model = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0
    )

    chain = prompt | model

    response = chain.invoke({
        "context": context,
        "question": state["question"]
    })

    return {"answer": response.content}


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def run_rag(question: str) -> dict:
    workflow = build_rag_graph()

    result = workflow.invoke({
        "question": question,
        "context": [],
        "answer": ""
    })

    sources = []

    for document in result["context"]:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page")

        item = {
            "source": str(source),
            "page": page
        }

        if item not in sources:
            sources.append(item)

    return {
        "answer": result["answer"],
        "sources": sources,
        "retrieved_chunks": len(result["context"])
    }