import os

from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0):
    """Return a chat LLM, picked by provider priority.

    1. Ollama   — used when OLLAMA_MODEL is set (local model, no API key needed).
    2. OpenAI   — used when OPENAI_API_KEY is set.
    3. Gemini   — used when GEMINI_API_KEY is set.
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
        "No LLM backend configured. Set one of OLLAMA_MODEL, OPENAI_API_KEY, "
        "or GEMINI_API_KEY in .env."
    )
