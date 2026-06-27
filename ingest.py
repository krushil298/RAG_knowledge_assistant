"""
ingest.py  —  PHASE 1 of RAG: build the knowledge index (offline, run once)

Pipeline:  load documents -> split into chunks -> embed each chunk -> store in Chroma
You run this whenever your source documents change.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # reads GOOGLE_API_KEY from the .env file

DOCS_DIR = "docs"        # folder containing your source .txt files
DB_DIR = "chroma_db"     # folder where the vector index is persisted


def main():
    # Fail early with a helpful message if the API key is missing.
    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("Error: GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")

    # The docs folder must exist and contain something to index.
    if not os.path.isdir(DOCS_DIR):
        sys.exit(f"Error: no '{DOCS_DIR}/' folder found. Create it and add your .txt files.")

    # 1. LOAD ---------------------------------------------------------------
    # Read every .txt file in docs/ into LangChain Document objects.
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s)")

    if not documents:
        sys.exit(f"Error: no .txt files found in '{DOCS_DIR}/'. Add documents and try again.")

    # 2. CHUNK --------------------------------------------------------------
    # A whole document is too big and too unfocused to embed as one vector.
    # We split it into ~1000-character chunks, with 150 chars of overlap so a
    # sentence cut at a boundary still appears whole in a neighbouring chunk.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # 3. EMBED + STORE ------------------------------------------------------
    # Each chunk is turned into a vector (a list of numbers capturing meaning).
    # Chroma stores the vectors + the original text + metadata, and persists
    # it all to disk so we don't have to re-embed every time we ask a question.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    print(f"Embedded and stored {len(chunks)} chunks in ./{DB_DIR}")


if __name__ == "__main__":
    main()
