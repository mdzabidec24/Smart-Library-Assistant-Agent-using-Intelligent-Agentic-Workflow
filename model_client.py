from langchain_ollama import OllamaLLM

def get_llm():
    # llama3.2 for 9th kid GK, low temperature to avoid overlap with library data
    return OllamaLLM(model="llama3.2", temperature=0.2)