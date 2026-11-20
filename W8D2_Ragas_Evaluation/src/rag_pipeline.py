import os
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "knowledge_base.txt")

MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"


def build_rag(chunk_size=500, k=2, collection="rag_collection"):

    db_path = os.path.join(BASE_DIR, "chroma_db", collection)

    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name=collection
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    llm = ChatOllama(
        model=MODEL,
        temperature=0
    )

    return retriever, llm


def generate_answer(question, retriever, llm):

    documents = retriever.invoke(question)

    context = "\n".join(
        document.page_content for document in documents
    )

    prompt = f"""
    Answer the question using ONLY the supplied context.

    Context:
    {context}

    Question:
    {question}

    Give a clear and accurate answer.
    If the answer is unavailable in the context, say so.
    """

    response = llm.invoke(prompt)

    answer = response.content

    contexts = [
        document.page_content for document in documents
    ]

    return answer, contexts