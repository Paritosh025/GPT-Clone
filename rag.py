from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_chroma import Chroma # for creating a vector store that allows for efficient similarity searches and retrieval of relevant documents based on their embeddings. It is used to store and manage the embeddings of uploaded documents, enabling the chatbot to perform RAG (Retrieval-Augmented Generation) operations.
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document # for representing documents in a structured format, including their content and metadata. It is used to create document objects that can be added to the vector store for retrieval and processing during RAG operations.
from langchain_text_splitters import RecursiveCharacterTextSplitter # for chunking the text into smaller pieces for better retrieval and processing. It splits the text into chunks of a specified size with a certain overlap between chunks to ensure context is preserved across chunks.

from pypdf import PdfReader
import docx2txt


Path("uploads").mkdir(exist_ok=True) # store in uploads folder if it doesn't exist
Path("chroma_db").mkdir(exist_ok=True) # store in chroma_db folder if it doesn't exist, use local storage to store the information about the uploaded documents and their embeddings. This allows for efficient retrieval of relevant document content during RAG operations without relying on external databases or cloud storage.


# Use a separate collection because existing vectors may have been created with a different provider.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma( 
    collection_name="agentic_chatbot_docs_openai",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)


# reads file text based on the file type (PDF, DOCX, TXT, MD, PY, or CSV) and returns the extracted text. It uses appropriate libraries 
# to handle different file formats and raises an error for unsupported file types.
def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"

        return text

    if suffix == ".docx":
        return docx2txt.process(file_path)

    if suffix in [".txt", ".md", ".py", ".csv"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.")



# function to create a vector store from the uploaded document. It reads the text from the file, splits it into chunks, creates Document objects 
# with metadata, and adds them to the Chroma vector store for efficient retrieval during RAG operations.
def add_document_to_rag(file_path: str, thread_id: str):
    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError("No text could be extracted from this file.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    docs: List[Document] = [
        Document(
            page_content=chunk,
            metadata={
                "thread_id": thread_id,
                "source": Path(file_path).name
            }
        )
        for chunk in chunks
    ]

    vectorstore.add_documents(docs)

    return {
        "filename": Path(file_path).name,
        "chunks": len(docs)
    }



# function to retrieve relevant content from the uploaded documents based on a query. It performs a similarity search in the Chroma vector store 
# using the provided query and thread ID, returning the most relevant document chunks along with their sources. If no relevant content is found, it returns a message indicating that.
def retrieve_from_rag(query: str, thread_id: str, k: int = 4) -> str:
    docs = vectorstore.similarity_search(
        query,
        k=k,
        filter={"thread_id": thread_id}
    )

    if not docs:
        return "No relevant uploaded document content found."

    results = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "uploaded document")
        results.append(
            f"[Source {i}: {source}]\n{doc.page_content}"
        )

    return "\n\n".join(results)