import os

from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0):
    """
    Return a chat LLM using the first available backend.

    Priority:
    1. Ollama
    2. OpenAI
    3. Gemini
    """

    ollama_model = os.getenv("OLLAMA_MODEL")
    if ollama_model:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=ollama_model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=openai_api_key,
            temperature=temperature,
        )

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            google_api_key=gemini_api_key,
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM backend configured. Set one of "
        "OLLAMA_MODEL, OPENAI_API_KEY, or GEMINI_API_KEY in .env."
    )


def get_embeddings():
    """
    Return an embeddings model using the same provider priority.

    Priority:
    1. Ollama
    2. OpenAI
    3. Gemini
    """

    ollama_model = os.getenv("OLLAMA_MODEL")
    if ollama_model:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            api_key=openai_api_key,
        )

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004"),
            google_api_key=gemini_api_key,
        )

    raise RuntimeError(
        "No embeddings backend configured. Set one of "
        "OLLAMA_MODEL, OPENAI_API_KEY, or GEMINI_API_KEY in .env."
    )
