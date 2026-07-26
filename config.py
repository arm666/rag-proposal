import os

from dotenv import load_dotenv

load_dotenv()


def _ollama_keep_alive_seconds() -> int:
    """OLLAMA_KEEP_ALIVE is documented in .env as seconds (e.g. "1800" for
    30 minutes, "-1" to keep the model resident indefinitely).
    OllamaEmbeddings only accepts an int number of seconds for keep_alive
    (unlike ChatOllama, which also accepts duration strings like "30m"), so
    both use the same int value here for consistency."""
    return int(os.getenv("OLLAMA_KEEP_ALIVE", "1800"))


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
            keep_alive=_ollama_keep_alive_seconds(),
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
            keep_alive=_ollama_keep_alive_seconds(),
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
