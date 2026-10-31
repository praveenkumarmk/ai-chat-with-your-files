from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone

from InitializePinecone import PINE_CONE_API_KEY

pc = Pinecone(PINE_CONE_API_KEY)

index = pc.Index("docs-index")
embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
}

def ingest_file(file_path: str, doc_id: str, namespace: str) -> list[str]:
    # 1. Load: one Document per page where the format supports it (metadata includes 'page', 0-based)
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADERS.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext} (supported: {', '.join(LOADERS)})")
    pages = loader_cls(file_path).load()

    # 2. Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    # 3. Flat, useful metadata
    docs = [
        Document(
            page_content=chunk.page_content,
            metadata={
                "doc_id": doc_id,
                "source": file_path,
                "page": chunk.metadata.get("page", 0) + 1,  # 1-based for citations
                "chunk_index": i,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    # 4. Deterministic IDs so re-ingesting overwrites instead of duplicating
    ids = [f"{doc_id}#chunk-{i}" for i in range(len(docs))]

    # 5. Embed + upsert (batched by LangChain)
    store = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)
    store.add_documents(docs, ids=ids)
    return store