V# CourseMate AI

CourseMate AI is a terminal-based conversational RAG assistant for studying from local PDF notes. It loads PDFs from `data/`, chunks them with LangChain, embeds them with Mistral embeddings, stores them in a local ChromaDB index, and answers questions with source-aware citations using Mistral chat models.

## Features

- PDF ingestion from the local `data/` directory
- Recursive text chunking with page metadata preservation
- Persistent ChromaDB vector store in `chroma_db/`
- Mistral `mistral-embed` embeddings
- Conversational question answering with chat history
- History-aware query reformulation for follow-up questions
- Source snippets and page references for retrieved context
- Simple terminal commands for reset, source inspection, and exit

## Project Structure

```text
.
|-- main.py                    # CLI entry point and knowledge-base bootstrap
|-- requirements.txt           # Python dependencies
|-- data/                      # Put study PDFs here
|-- chroma_db/                 # Persisted Chroma vector database
`-- src/
    |-- ingest.py              # PDF loading and chunking
    |-- vectorstore.py         # ChromaDB and Mistral embedding management
    |-- conversational_rag.py  # Multi-turn conversational RAG chain
    `-- ragchain.py            # Single-turn RAG chain utility
```

## Requirements

- Python 3.10+
- A Mistral API key
- PDF study material placed in `data/`

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

4. Add your course PDFs to the `data/` folder:

```text
data/
|-- Computer Networks Notes.pdf
`-- SQL Handwrritten Notes.pdf
```

## Usage

Run the assistant:

```powershell
python main.py
```

On first run, CourseMate AI will:

1. Check for PDFs in `data/`
2. Load and split the PDFs into overlapping chunks
3. Generate embeddings with Mistral
4. Store the vectors in `chroma_db/`
5. Start an interactive chat session

Example prompts:

```text
Student: Explain normalization in DBMS.
Student: What is TCP congestion control?
Student: Summarize this topic in exam-answer format.
```

Available commands:

```text
/reset    Clear the current conversation history
/sources  Show retrieved source snippets from the previous answer
/exit     Quit the application
```

## How It Works

The app uses a two-stage conversational RAG pipeline:

1. `src/ingest.py` loads PDF pages and splits them into chunks while keeping source file and page metadata.
2. `src/vectorstore.py` embeds chunks with `mistral-embed` and stores them in ChromaDB.
3. `src/conversational_rag.py` reformulates follow-up questions into standalone questions, retrieves relevant chunks, and asks Mistral to answer only from the retrieved context.
4. `main.py` ties ingestion, retrieval, and the terminal chat loop together.

The assistant is prompted to avoid outside knowledge and to respond with citations such as:

```text
[Source: Computer Networks Notes.pdf, Page 12]
```

## Re-indexing Notes

If `chroma_db/` already contains vectors, the app reuses the existing database instead of ingesting PDFs again. To rebuild the index after adding, removing, or changing PDFs, stop the app and remove the existing Chroma database:

```powershell
Remove-Item -Recurse -Force .\chroma_db
```

Then run:

```powershell
python main.py
```

## Troubleshooting

### Missing API Key

If you see:

```text
[ERROR] Missing MISTRAL_API_KEY in your .env file.
```

make sure `.env` exists in the project root and contains `MISTRAL_API_KEY`.

### No PDFs Found

If the app starts with an empty index, add `.pdf` files to `data/` and rerun the app.

### Dependency Import Errors

Install or refresh dependencies:

```powershell
pip install -r requirements.txt
```

If `langchain_classic` is missing, install the LangChain classic package used by `src/conversational_rag.py`:

```powershell
pip install langchain-classic
```

## Notes

- Keep `.env`, `.venv/`, and generated vector stores out of version control.
- Answers are only as good as the text extracted from the PDFs.
- Scanned handwritten PDFs may need OCR before ingestion if text extraction returns poor results.
