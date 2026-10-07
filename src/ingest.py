import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    """
    Handles extraction, metadata preservation, and semantic chunking
    for academic PDF documents.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Priority: Double newline (paragraphs) -> Single newline (lines) -> Spaces (words) -> Chars
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_single_pdf(self, file_path: str) -> List[Document]:
        """Loads a single PDF and preserves page-level metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target PDF not found at: {file_path}")
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return docs

    def load_directory(self, dir_path: str) -> List[Document]:
        """Loads all PDFs in a directory."""
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Target directory not found at: {dir_path}")
            
        loader = PyPDFDirectoryLoader(dir_path)
        docs = loader.load()
        return docs

    def process_and_chunk(self, docs: List[Document]) -> List[Document]:
        """
        Splits documents into overlapping chunks while propagating metadata.
        """
        chunks = self.splitter.split_documents(docs)
        
        # Clean metadata: standardize file names for clear downstream citations
        for chunk in chunks:
            raw_source = chunk.metadata.get("source", "Unknown")
            chunk.metadata["file_name"] = os.path.basename(raw_source)
            # PyPDF uses 0-based index; convert to 1-based index for human-readable citation
            if "page" in chunk.metadata:
                chunk.metadata["page_number"] = chunk.metadata["page"] + 1
        
        return chunks


if __name__ == "__main__":
    # Quick module smoke-test
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    # Example dry run with dummy Document
    sample_doc = [
        Document(
            page_content="CourseMate AI Architecture.\n\nSection 1: Data Ingestion.\n"
                         "PyPDF parses PDF text while retaining page numbers. "
                         "Recursive character splitting ensures paragraphs remain intact.",
            metadata={"source": "data/sample_lecture.pdf", "page": 0}
        )
    ]
    
    chunks = processor.process_and_chunk(sample_doc)
    print(f"Total Chunks Created: {len(chunks)}")
    print(f"Sample Chunk Metadata: {chunks[0].metadata}")
    print(f"Sample Chunk Content:\n{chunks[0].page_content}")