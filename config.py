import functools
import os

from dotenv import load_dotenv

load_dotenv()


def neo4j_enabled() -> bool:
    """Graph RAG storage backend: Neo4j when NEO4J_URI is set, otherwise the
    local NetworkX+JSON pipeline (see ingestion/graph_index.py and
    retrieval/graph_rag.py) — same provider-selection pattern as get_llm()."""
    return bool(os.getenv("NEO4J_URI"))


def neo4j_database() -> str:
    return os.getenv("NEO4J_DATABASE", "neo4j")


@functools.lru_cache(maxsize=1)
def get_neo4j_driver():
    """Return a process-wide singleton Neo4j driver. The driver manages its
    own connection pool and is meant to be created once and reused (unlike
    get_llm()/get_embeddings(), which are cheap to reconstruct per call) —
    only call this when neo4j_enabled() is True."""
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password))


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
            # Without this, Ollama falls back to its runtime default context
            # window (4096 tokens) regardless of what the model itself
            # supports. graph_index.py's extraction chunks (6000 chars) plus
            # a dense triplet-JSON response can blow past 4096 combined
            # tokens, silently truncating the model's output mid-JSON — the
            # response never errors, it just stops mid-object, so
            # _parse_triplets can't find a closing bracket and the whole
            # chunk quietly contributes zero triplets. 8192 covers a 6000-
            # char chunk (~1500-2000 tokens) plus room for a large triplet
            # array; raise OLLAMA_NUM_CTX further for bigger chunk sizes.
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
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
