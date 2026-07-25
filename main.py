from config import get_llm


def main():
    llm = get_llm()
    print(f"Using LLM backend: {llm.__class__.__name__}")


if __name__ == "__main__":
    main()
