"""
Generator: calls Groq's LLM (via langchain-groq) with the retrieved context.
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import settings

SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the
provided context. If the answer is not contained in the context, say you don't know
rather than guessing. Always cite the [source] tags from the context you used."""

_llm_instance = None


def get_llm() -> ChatGroq:
    global _llm_instance
    if _llm_instance is None:
        settings.validate()
        _llm_instance = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0.2,
        )
    return _llm_instance


def generate_answer(question: str, context: str) -> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
