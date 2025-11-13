# AI Concepts Practice

Streamlit app with LLM (semantic) scoring + ROUGE-L similarity and a blended score.

## Setup
```powershell
python -m venv chatbot-env
.\chatbot-env\Scripts\Activate.ps1
pip install --upgrade pip wheel
pip install -r requirements.txt

## Ollama model

Pull and run the model locally:

ollama pull llama3.2:3b
ollama serve


Then run the app:

streamlit run chatbot.py
