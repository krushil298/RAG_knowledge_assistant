"""
query.py  —  PHASE 2 of RAG: answer questions (online, run any time)

Pipeline:  embed the question -> find nearest chunks in Chroma -> stuff them
into the prompt -> the LLM answers ONLY from that context -> show sources.

Usage:
    python query.py                  # interactive mode
    python query.py "your question"  # one-shot mode (answer and exit)
"""

import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

load_dotenv()
DB_DIR = "chroma_db"

# The grounding prompt. The two instructions in CAPS are what stop the model
# from inventing answers — critical in a clinical setting.
PROMPT_TEMPLATE = """You are a careful assistant for clinical research staff.
Answer the QUESTION using ONLY the CONTEXT below.
If the context does not contain the answer, reply exactly:
"I don't have enough information to answer that."
Do not use any outside knowledge.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def answer(retriever, llm, question):
    """Retrieve relevant chunks, generate a grounded answer, and show sources."""
    # RETRIEVE
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # GENERATE (grounded)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)

    print("\nANSWER:\n" + response.content)
    print("\nRETRIEVED FROM:")
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] {d.metadata.get('source', 'unknown')}")


def main():
    # Fail early with a helpful message if the API key is missing.
    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("Error: GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")

    # The index must exist before we can query it.
    if not os.path.isdir(DB_DIR):
        sys.exit(f"Error: no index found at ./{DB_DIR}. Run `python ingest.py` first.")

    # Re-open the index we built in ingest.py. We must use the SAME embedding
    # model here, or the question vector won't live in the same space as the chunks.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # Retriever = "embed the query, return the k closest chunks".
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # temperature=0 -> deterministic, factual answers (no creative drift).
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # One-shot mode: a question passed on the command line is answered, then exit.
    if len(sys.argv) > 1:
        answer(retriever, llm, " ".join(sys.argv[1:]))
        return

    # Interactive mode.
    print("RAG assistant ready. Ask a question (type 'quit' to exit).")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break
        answer(retriever, llm, question)


if __name__ == "__main__":
    main()
