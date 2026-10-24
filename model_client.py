from langchain_ollama import OllamaLLM
def get_llm():
    return OllamaLLM(model="llama3.2", temperature=0.2)