import os
import sys
import glob
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("MISTRAL_API_KEY"):
    print(" [ERROR] Missing MISTRAL_API_KEY in your .env file.")
    sys.exit(1)

from src.ingest import DocumentProcessor
from src.vectorstore import VectorStoreManager
from src.conversational_rag import ConversationalCourseMate

def bootstrap_knowledge_base(data_dir: str = "./data", persist_dir: str = "./chroma_db") -> VectorStoreManager:
    """
    Checks for PDFs in data_dir. If the database is empty or new documents exist,
    it parses, chunks, and indexes them into ChromaDB.
    """
    os.makedirs(data_dir, exist_ok=True)
    vector_mgr = VectorStoreManager(persist_directory=persist_dir)

    existing_chunks_count = vector_mgr.vector_store._collection.count()
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))

    print(f"\n" + "="*60)
    print(f" CourseMate AI - Knowledge Base Initializer")
    print(f"="*60)
    print(f"[*] Found {len(pdf_files)} PDF(s) in '{data_dir}'")
    print(f"[*] Existing vectors in ChromaDB: {existing_chunks_count}")

    if existing_chunks_count == 0:
        if not pdf_files:
            print(f"[!] Warning: No PDF files found in '{data_dir}'.")
            print(f"    Please drop your lecture notes/books into '{data_dir}' and rerun.")
            print(f"    Proceeding with an empty index...\n")
        else:
            print(f"[*] Ingesting and chunking documents from '{data_dir}'...")
            processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
            documents = processor.load_directory(data_dir)
            chunks = processor.process_and_chunk(documents)
            
            print(f"[*] Indexing {len(chunks)} chunks into ChromaDB...")
            vector_mgr.index_documents(chunks)
            print("[+] Ingestion & indexing completed successfully!\n")
    else:
        print("[+] Existing persistent vector store loaded.\n")

    return vector_mgr


def main():
    # 1. Initialize Vector Database & Ingestion
    vector_manager = bootstrap_knowledge_base()
    
    # 2. Configure Retriever (k=4 closest relevant chunks)
    retriever = vector_manager.get_retriever(search_type="similarity", k=4)
    
    # 3. Instantiate Conversational Assistant
    assistant = ConversationalCourseMate(
        retriever=retriever,
        model_name="mistral-large-latest",
        temperature=0.0
    )

    # 4. Interactive Terminal Loop
    print("="*60)
    print(" CourseMate AI — Academic Study Assistant Ready")
    print(" Commands:")
    print("   /reset   - Clear conversational history")
    print("   /sources - Show full retrieved snippets from the last turn")
    print("   /exit    - Quit application")
    print("="*60 + "\n")

    last_sources = []

    while True:
        try:
            user_input = input("\nStudent: ").strip()

            if not user_input:
                continue

            # Command Handlers
            if user_input.lower() in ["/exit", "exit", "quit"]:
                print("\nCourseMate AI: Good luck with your studies! Goodbye.\n")
                break

            elif user_input.lower() == "/reset":
                assistant.clear_memory()
                last_sources = []
                print("[*] Conversation history cleared.")
                continue

            elif user_input.lower() == "/sources":
                if not last_sources:
                    print("[!] No sources available from previous query.")
                else:
                    print("\n--- Retrieved Sources for Previous Response ---")
                    for idx, src in enumerate(last_sources, 1):
                        print(f"[{idx}] File: {src['file']} | Page: {src['page']}")
                        print(f"    Excerpt: {src['snippet']}\n")
                continue

            # Run RAG Query
            print("\nCourseMate AI is thinking...", end="\r")
            result = assistant.chat(user_input)
            last_sources = result["sources"]

            # Clear 'thinking' status and print output
            print(" " * 30, end="\r")
            print(f"CourseMate AI:\n{result['answer']}")

            # Print concise source attribution line
            if result["sources"]:
                cited_pages = {f"{s['file']} (p. {s['page']})" for s in result["sources"]}
                print(f"\n Sources referenced: {', '.join(cited_pages)}")

        except KeyboardInterrupt:
            print("\n\nSession terminated by user.")
            break
        except Exception as e:
            print(f"\n[!] An error occurred: {str(e)}")


if __name__ == "__main__":
    main()