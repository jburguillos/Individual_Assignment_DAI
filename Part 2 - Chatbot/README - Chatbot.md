# 🤖 AI Concepts Practice

Streamlit app with **LLM (semantic)** scoring, **ROUGE-L** similarity, and a **blended score** for evaluating your understanding of AI/ML concepts.

Developed for the *Designing Artificial Intelligence – Part 2* assignment.

---

## ⚙️ Setup

```powershell
python -m venv chatbot-env
.\chatbot-env\Scripts\Activate.ps1
pip install --upgrade pip wheel
pip install -r requirements.txt
🧠 Ollama Model
This app uses a local Ollama model as the evaluator for semantic grading.

1. Install Ollama
Download and install from 👉 https://ollama.com/download
(available for Windows, macOS, and Linux)

2. Pull the required model
The default model is:

bash
Copiar código
ollama pull llama3.2:3b
If you prefer another model, edit this line in chatbot.py:

python
Copiar código
llm = ChatOllama(model="llama3.2:3b", temperature=0)
and pull your chosen model with:

bash
Copiar código
ollama pull <model-name>
3. Run Ollama in the background
Make sure the Ollama service is running before launching Streamlit:

bash
Copiar código
ollama serve
▶️ Run the App
After activating your environment and running Ollama:

bash
Copiar código
streamlit run chatbot.py
Open the local URL displayed in the terminal (usually http://localhost:8501).

🧮 Scoring Overview
Metric	Purpose	Scale
LLM (semantic)	Evaluates conceptual accuracy and understanding	0 – 100
ROUGE-L	Measures text overlap and structural similarity (Longest Common Subsequence)	0 – 1 (also shown as %)
Blended	Weighted combination (default 60 % ROUGE-L + 40 % LLM)	0 – 100

🧠 ROUGE-L rewards phrase and structure matching,
while the LLM focuses on meaning and conceptual correctness.

💡 Example Workflow
Run ollama serve

Start the Streamlit app:

bash
Copiar código
streamlit run chatbot.py
Answer the question shown in the UI

Receive evaluation results:

Semantic score (LLM)

ROUGE-L similarity

Blended total

Feedback explaining strengths and weaknesses

Rate the evaluation (“Useful”, “Too strict”, etc.)

Continue to the next question or review your results table.

