import os
from typing import List
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


class ConversationalCourseMate:
    """
    Multi-turn Conversational RAG assistant with query contextualization,
    source citations, and in-memory chat history.
    """
    def __init__(
        self,
        retriever,
        model_name: str = "mistral-large-latest",
        temperature: float = 0.0
    ):
        self.retriever = retriever
        self.llm = ChatMistralAI(
            model=model_name,
            temperature=temperature,
            mistral_api_key=os.getenv("MISTRAL_API_KEY")
        )
        
        # Local session memory store
        self.chat_history: List[BaseMessage] = []
        
        # Build the sub-chains
        self.rag_chain = self._build_conversational_chain()

    def _build_conversational_chain(self):
        # -------------------------------------------------------------
        # Stage 1: Contextualize Question (Query Reformulation)
        # -------------------------------------------------------------
        contextualize_q_system_prompt = (
            "Given a chat history and the latest student question which might reference "
            "context in the chat history, formulate a standalone question that can be "
            "understood without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=self.retriever,
            prompt=contextualize_q_prompt
        )

        # -------------------------------------------------------------
        # Stage 2: Question-Answering with Grounding & Inline Citations
        # -------------------------------------------------------------
        qa_system_prompt = (
            "You are CourseMate AI, an expert academic study assistant.\n"
            "STRICT GROUNDING RULES:\n"
            "1. Answer the question using ONLY the provided context materials below.\n"
            "2. Do NOT speculate or bring in outside knowledge.\n"
            "3. If the context does not contain the answer, state clearly: "
            "'I cannot find the answer to this in the provided study materials.'\n"
            "4. Provide inline citations for every assertion referencing the source document "
            "and page number (e.g., [Source: filename.pdf, Page X]).\n\n"
            "Context Materials:\n{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # Stuff documents chain combines retrieved chunks into the {context} variable
        question_answer_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=qa_prompt
        )

        # -------------------------------------------------------------
        # Final Unified Chain
        # -------------------------------------------------------------
        return create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=question_answer_chain
        )

    def chat(self, user_message: str) -> dict:
        """
        Executes a single conversational turn, updates internal memory,
        and returns the answer and retrieved source documents.
        """
        response = self.rag_chain.invoke({
            "input": user_message,
            "chat_history": self.chat_history
        })
        
        answer = response["answer"]
        retrieved_docs = response.get("context", [])
        
        # Append turn to history
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=answer))
        
        return {
            "answer": answer,
            "sources": [
                {
                    "file": doc.metadata.get("file_name", "Unknown"),
                    "page": doc.metadata.get("page_number", doc.metadata.get("page", "N/A")),
                    "snippet": doc.page_content[:150] + "..."
                }
                for doc in retrieved_docs
            ]
        }

    def clear_memory(self):
        """Resets the conversation history."""
        self.chat_history = []