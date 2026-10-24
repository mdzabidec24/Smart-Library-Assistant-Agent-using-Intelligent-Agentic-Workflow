# 📚 Smart Library Assistant Agent using Intelligent Agentic Workflow

An AI-powered library assistant that helps students find, search, and get recommendations for books using Ollama local LLM and agentic workflow.

### What is Agentic Workflow?
Instead of one AI, we use multiple specialized agents working as a team:

1. Intent Agent (model_client.py) - Understands user query using Ollama
2. Search Agent - Searches books in data/library.csv
3. Recommend Agent (tools/) - Recommends best books based on user interest
4. Response Agent - Gives final friendly answer

This is called Agentic Workflow - Agents pass work to each other like humans in office!

### Why Ollama?
- 100% Free - No API key needed
- Runs locally - No internet needed after download
- Private - Your library data never goes to cloud
- Fast - Uses llama3.2 model

### Features
- Search books using normal English
- Smart recommendations based on department/year
- Checks availability and shelf number
- Conversational memory
- Offline capable with Ollama
- Built with Streamlit

### Project Structure
- app.py - Main Streamlit UI
- model_client.py - Ollama connection and intent detection
- data/library.csv - Books database
- tools/ - Search and recommendation logic
- requirements.txt - Dependencies
- .gitignore - Ignores venv

### How to Run Locally

Step 1: Install Ollama from https://ollama.com

Step 2: Pull model
ollama pull llama3

Step 3: Clone repo
git clone https://github.com/mdzabidec24/Smart-Library-Assistant-Agent-using-Intelligent-Agentic-Workflow.git

Step 4: Create venv and install
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Step 5: Run
ollama run llama3 (keep this running in one terminal)
streamlit run app.py (run in second terminal)

### Example Queries
- Find book on Data Structures
- I am CSE 2nd year, recommend Python books
- Is Let Us C available?
- Suggest ML books for beginners

### Tech Stack
- Frontend: Streamlit
- LLM: Ollama (llama3)
- Language: Python
- Data: CSV

### Author
B.Mohammed Zabi - CSE
GitHub: mdzabidec24
