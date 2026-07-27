import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.core.config import SETTINGS

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        groq_key = SETTINGS.GROQ_API_KEY
        if groq_key:
            self.client = ChatGroq(
                api_key=groq_key,
                model_name='llama-3.1-8b-instant',
                temperature=0.3,
                max_tokens=1000
            )
        else:
            self.client = None
            logger.warning("GROQ_API_KEY missing in SETTINGS. LLMService disabled.")
            
        self.model_name = "llama-3.1-8b-instant"

    async def stream_answer(
        self, 
        query: str, 
        citations: List[Dict[str, Any]], 
        chat_history: Optional[List[Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously streams the LLM answer token-by-token while factoring in
        conversational memory and retrieved document citations.
        """
        if not self.client:
            raise ValueError("LLM Client not initialized. Check Groq API Key.")

        # 1. Format Context from Citations
        context_text = ""
        for idx, cite in enumerate(citations, start=1):
            source = cite.get("source_file", "Unknown")
            page = cite.get("page_number", "N/A")
            snippet = cite.get("text_snippet", "").strip()
            context_text += (
                f"--- Citation {idx} (File: {source}, Page: {page}) ---\n"
                f"{snippet}\n"
            )

        # 2. Prompts Setup
        system_prompt = (
            "You are an expert AI assistant specializing in document analysis.\n"
            "Your core task is to answer the user's question STRICTLY based on the provided Context.\n\n"
            "STRICT RULES:\n"
            "1. Do NOT use any outside knowledge or make assumptions beyond the provided Context.\n"
            "2. If the answer cannot be determined from the Context, explicitly state: "
            "'I do not have enough information in the provided documents to answer this question.'\n"
            "3. Keep your response clear, concise, and directly factual to the provided text."
        )

        user_prompt = (
            f"CONTEXT INFORMATION:\n"
            f"{context_text}\n"
            f"---------------------\n"
            f"USER QUESTION: {query}\n\n"
            f"ANSWER:"
        )

        # 3. Build Base Message Stack
        messages = [SystemMessage(content=system_prompt)]

        # 4. Safely Inject Conversational Memory
        if chat_history:
            for msg in chat_history:
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                
                if role and content:
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))

        # 5. Append Final Prompt with Retrieved Context
        messages.append(HumanMessage(content=user_prompt))

        # 6. Stream
        try:
            async for chunk in self.client.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            yield "\n[An error occurred while generating the response.]"

    def generate_answer(
        self, 
        query: str, 
        citations: List[Dict[str, Any]], 
        chat_history: Optional[List[Any]] = None
    ) -> str:
        """
        Synchronous fallback method for non-streaming calls.
        """
        if not self.client:
            raise ValueError("LLM Client not initialized. Check Groq API Key.")

        context_text = ""
        for idx, cite in enumerate(citations, start=1):
            source = cite.get("source_file", "Unknown")
            page = cite.get("page_number", "N/A")
            snippet = cite.get("text_snippet", "").strip()
            context_text += (
                f"--- Citation {idx} (File: {source}, Page: {page}) ---\n"
                f"{snippet}\n"
            )

        system_prompt = (
            "You are an expert AI assistant specializing in document analysis.\n"
            "Your core task is to answer the user's question STRICTLY based on the provided Context.\n\n"
            "STRICT RULES:\n"
            "1. Do NOT use any outside knowledge or make assumptions beyond the provided Context.\n"
            "2. If the answer cannot be determined from the Context, explicitly state: "
            "'I do not have enough information in the provided documents to answer this question.'\n"
            "3. Keep your response clear, concise, and directly factual to the provided text."
        )

        user_prompt = (
            f"CONTEXT INFORMATION:\n"
            f"{context_text}\n"
            f"---------------------\n"
            f"USER QUESTION: {query}\n\n"
            f"ANSWER:"
        )

        messages = [SystemMessage(content=system_prompt)]

        if chat_history:
            for msg in chat_history:
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                
                if role and content:
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=user_prompt))

        try:
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "An error occurred while generating the response."