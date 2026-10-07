import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def format_docs_with_sources(docs: List[Document]) -> str:
    """
    Transforms retrieved LangChain Documents into a structured,
    source-attributed context string for the prompt.
    """
    formatted_chunks = []
    for doc in docs:
        file_name = doc.metadata.get("file_name", "Unknown Document")
        page_num = doc.metadata.get("page_number", doc.metadata.get("page", "N/A"))
        
        # Structure each chunk clearly with explicit boundary markers
        chunk_text = (
            f"--- START SOURCE: {file_name} (Page {page_num}) ---\n"
            f"{doc.page_content.strip()}\n"
            f"--- END SOURCE ---"
        )
        formatted_chunks.append(chunk_text)
    
    return "\n\n".join(formatted_chunks)


class CourseMateRAG:
    """
    Orchestrates the end-to-end RAG pipeline using Mistral AI and LangChain LCEL.
    """
    def __init__(
        self,
        retriever,
        model_name: str = "mistral-large-latest",
        temperature: float = 0.0
    ):
        self.retriever = retriever
        
        # Initialize Mistral AI LLM
        # temperature=0.0 minimizes creative drift and maximizes deterministic adherence
        self.llm = ChatMistralAI(
            model=model_name,
            temperature=temperature,
            mistral_api_key=os.getenv("MISTRAL_API_KEY")
        )
        
        # Define the system and user prompt with strict guardrails
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are CourseMate AI, an expert academic study assistant designed to help students "
                "understand their lecture notes, textbooks, and research papers.\n\n"
                "STRICT GROUNDING GUIDELINES:\n"
                "1. Answer the student's question using ONLY the facts explicitly provided in the context below.\n"
                "2. Do NOT extrapolate, speculate, or introduce external knowledge outside of the context.\n"
                "3. If the context does not contain enough information to answer the question, state clearly: "
                "'I cannot find the answer to this in the provided study materials.'\n"
                "4. Every factual assertion must be attributed with an inline citation referring to its source and page, "
                "e.g., [Source: notes.pdf, Page 12].\n"
                "5. Provide structured, clear, and well-reasoned answers using Markdown (lists, bold key terms)."
            ),
            (
                "human",
                "Context Study Materials:\n"
                "{context}\n\n"
                "Student Question: {question}\n\n"
                "Academic Response:"
            )
        ])
        
        self.chain = self._build_chain()

    def _build_chain(self):
        """
        Builds the unified LCEL (LangChain Expression Language) pipeline.
        """
        # LCEL execution flow:
        # 1. Takes {"question": str}
        # 2. Runs retriever to get docs -> formats docs with sources -> assigns to 'context'
        # 3. Passes raw question to 'question'
        # 4. Injects context and question into prompt
        # 5. Sends formatted prompt to Mistral LLM
        # 6. Parses AIMessage output to plain string
        chain = (
            {
                "context": self.retriever | format_docs_with_sources,
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        return chain

    def query(self, question: str) -> str:
        """Executes a query through the RAG chain."""
        return self.chain.invoke(question)


if __name__ == "__main__":
    # Unit-test prompt formatting and context generation
    sample_docs = [
        Document(
            page_content="Supervised learning uses labeled training datasets to map input features to output targets.",
            metadata={"file_name": "machine_learning_101.pdf", "page_number": 3}
        ),
        Document(
            page_content="Common supervised learning algorithms include Linear Regression, Logistic Regression, and SVMs.",
            metadata={"file_name": "machine_learning_101.pdf", "page_number": 4}
        )
    ]
    
    formatted_context = format_docs_with_sources(sample_docs)
    print("--- Generated Context Block ---")
    print(formatted_context)