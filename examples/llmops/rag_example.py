"""
🔥 LLM with RAG Example (2024-2025 Trend!)
Retrieval Augmented Generation using LangChain and Qdrant
"""

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
import os


def setup_rag_system():
    """
    Setup a complete RAG system with:
    - Document loading and chunking
    - Vector embeddings with OpenAI
    - Qdrant vector database
    - LangChain QA chain
    """

    # 1. Load documents
    print("📚 Loading documents...")
    loader = TextLoader("your_documents.txt")
    documents = loader.load()

    # 2. Split documents into chunks
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)

    # 3. Create embeddings
    print("🎯 Creating embeddings...")
    embeddings = OpenAIEmbeddings()

    # 4. Setup Qdrant vector database
    print("🗃️ Setting up Qdrant vector database...")
    qdrant_client = QdrantClient(url="http://localhost:6333")

    vectorstore = Qdrant.from_documents(
        texts,
        embeddings,
        url="http://localhost:6333",
        collection_name="my_documents",
        prefer_grpc=False
    )

    # 5. Create QA chain
    print("🔗 Creating QA chain...")
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}
        ),
        return_source_documents=True
    )

    return qa_chain


def query_documents(qa_chain, question: str):
    """
    Query the RAG system with a question
    """
    print(f"\n❓ Question: {question}")

    result = qa_chain({"query": question})

    print(f"\n✅ Answer: {result['result']}")
    print(f"\n📄 Sources:")
    for i, doc in enumerate(result['source_documents']):
        print(f"  {i+1}. {doc.page_content[:100]}...")


if __name__ == "__main__":
    # Setup RAG system
    qa_chain = setup_rag_system()

    # Example queries
    questions = [
        "What is MLOps?",
        "How do you deploy ML models?",
        "What are the best practices for model monitoring?"
    ]

    for question in questions:
        query_documents(qa_chain, question)
        print("\n" + "="*80 + "\n")
