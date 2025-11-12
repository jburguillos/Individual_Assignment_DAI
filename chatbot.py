""" import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.set_page_config(page_title="Chatbot")
st.title("Chatbot")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.session_state.messages.append(
    SystemMessage("Act like a GIS Professional with 20 years of experience.")
)

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


#creating the bar to input the question
prompt = st.chat_input("Ask me anything about the project")

#Did the user type something?
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    
        st.session_state.messages.append(HumanMessage(prompt))
        # st.session_state.messages.append({"role": "user", "content": prompt})

        # create the echo (response) and add it to the screen
    llm = ChatOllama(model="llama3.2:3b", temperature=2)

    result = llm.invoke(st.session_state.messages).content

    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append(AIMessage(result)) """

# chatbot.py (styled UI + ROUGE-L + blended score + robust JSON parsing)
import json, random, re, ast
from pathlib import Path
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------- THEME & UI CONFIG ----------------
st.set_page_config(page_title="AI Concepts Practice", page_icon="🤖", layout="centered")

# Custom CSS (palette + components)
PALETTE = {
    "space_cadet": "#1b264f",
    "marian_blue": "#274690",
    "true_blue":  "#576ca8",
    "jet":        "#302b27",
    "smoke":      "#f5f3f5",
    "accent":     "#47c6ff",   # subtle accent for highlights
    "success":    "#21c18f",
    "warning":    "#f7b500",
    "danger":     "#f55353",
}

st.markdown(
    f"""
    <style>
      :root {{
        --space-cadet: {PALETTE["space_cadet"]};
        --marian-blue: {PALETTE["marian_blue"]};
        --true-blue:  {PALETTE["true_blue"]};
        --jet:        {PALETTE["jet"]};
        --smoke:      {PALETTE["smoke"]};
        --accent:     {PALETTE["accent"]};
        --success:    {PALETTE["success"]};
        --warning:    {PALETTE["warning"]};
        --danger:     {PALETTE["danger"]};
      }}
      /* App background & typography */
      .stApp {{
        background: linear-gradient(180deg, var(--smoke) 0%, #ffffff 80%);
        color: var(--jet);
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
      }}

      /* Header badge */
      .hero {{
        background: linear-gradient(135deg, var(--space-cadet), var(--marian-blue));
        color: white;
        padding: 22px 24px;
        border-radius: 18px;
        box-shadow: 0 8px 20px rgba(27,38,79,0.18);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 14px;
      }}
      .hero h1 {{
        margin: 0 0 6px 0;
        font-size: 28px;
        letter-spacing: 0.2px;
      }}
      .hero p {{
        margin: 4px 0 0 0;
        opacity: 0.9;
      }}

      /* Card containers */
      .card {{
        background: #ffffff;
        border: 1px solid rgba(27,38,79,0.07);
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 6px 16px rgba(27,38,79,0.06);
        margin: 10px 0 14px;
      }}
      .card h3 {{
        margin-top: 0;
      }}

      /* Buttons */
      .stButton > button {{
        background: var(--marian-blue) !important;
        color: #fff !important;
        border: 1px solid rgba(0,0,0,0.05);
        padding: 10px 16px;
        border-radius: 12px !important;
        box-shadow: 0 3px 8px rgba(39,70,144,0.25);
        transition: transform 0.02s ease-in-out, box-shadow 0.2s;
        font-weight: 600;
      }}
      .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(39,70,144,0.32);
        background: #2f58bb !important;
      }}

      /* Primary submit in form (we'll wrap it in columns to stretch) */
      .block-container {{
        padding-top: 1.2rem;
        max-width: 920px;
      }}

      /* Text area */
      textarea, .stTextArea textarea {{
        border-radius: 12px !important;
        border: 1px solid rgba(27,38,79,0.18) !important;
        background: #fff !important;
      }}

      /* Sidebar */
      section[data-testid="stSidebar"] > div {{
        background: linear-gradient(180deg, #f6f7fb 0%, #ffffff 70%);
      }}
      .sidebar-card {{
        border: 1px solid rgba(27,38,79,0.06);
        background: #ffffff;
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 4px 10px rgba(27,38,79,0.06);
      }}
      .sidebar-card h4 {{
        margin: 0 0 8px 0;
      }}

      /* Metrics */
      [data-testid="stMetricValue"] {{
        color: var(--space-cadet);
        font-weight: 700;
      }}
      [data-testid="stMetricLabel"] {{
        color: rgba(48,43,39,0.75);
      }}

      /* Radio pills */
      div[role="radiogroup"] > label {{
        background: #fff;
        border: 1px solid rgba(27,38,79,0.14);
        padding: 6px 10px;
        border-radius: 999px;
        margin-right: 6px;
      }}
      div[role="radiogroup"] > label:hover {{
        border-color: var(--true-blue);
      }}

      /* Dataframe tweak */
      .stDataFrame div[data-testid="stHeader"] {{
        background: #f8f9fd;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="hero">
      <h1>🤖 Artificial Intelligence Concepts Practice</h1>
      <p>Designing Artificial Intelligence – Part 2 • 
         <b>Tip:</b> The LLM grades <i>semantic understanding</i>; ROUGE-L checks <i>textual overlap</i>.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Try to give the most accurate answer to the concept below. "
    "Higher ROUGE-L weight means more rigor on phrasing/structure, not just meaning."
)

# ---------------- ROUGE METRIC (optional) ----------------
@st.cache_resource
def _get_rouge_backends():
    """Try 'evaluate' first; if it fails, use 'rouge_score' directly (pure Python)."""
    backend = "none"
    eval_metric = None
    scorer = None
    try:
        import evaluate  # pip install evaluate rouge-score
        eval_metric = evaluate.load("rouge")
        backend = "evaluate"
        return {"backend": backend, "evaluate_metric": eval_metric, "rouge_scorer": None}
    except Exception:
        pass
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        backend = "rouge_score"
        return {"backend": backend, "evaluate_metric": None, "rouge_scorer": scorer}
    except Exception:
        return {"backend": "none", "evaluate_metric": None, "rouge_scorer": None}

_rouge = _get_rouge_backends()

def compute_rougeL(pred: str, ref: str) -> float | None:
    """Return ROUGE-L F1 in [0,1] or None if no backend is available."""
    try:
        if _rouge["backend"] == "evaluate" and _rouge["evaluate_metric"] is not None:
            res = _rouge["evaluate_metric"].compute(predictions=[pred], references=[ref])
            val = res.get("rougeL", res.get("rougeLsum", None))
            return float(val) if val is not None else None
        if _rouge["backend"] == "rouge_score" and _rouge["rouge_scorer"] is not None:
            # rouge_scorer expects (reference, prediction)
            scores = _rouge["rouge_scorer"].score(ref, pred)
            return float(scores["rougeL"].fmeasure)
        return None
    except Exception:
        return None

def blend_scores(rouge_l_0to1: float | None, llm_score_0to100: int | None,
                 w_rouge: float, w_llm: float) -> int | None:
    """Blend to [0..100]. If one component is missing, return the other."""
    if rouge_l_0to1 is None and llm_score_0to100 is None:
        return None
    if rouge_l_0to1 is None:
        return int(round(llm_score_0to100))
    if llm_score_0to100 is None:
        return int(round(rouge_l_0to1 * 100))
    return int(round(w_rouge * (rouge_l_0to1 * 100) + w_llm * llm_score_0to100))

# ---------------- HELPERS ----------------
@st.cache_data
def load_questions(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def pick_question(qa_list, asked_indices):
    remaining = [i for i in range(len(qa_list)) if i not in asked_indices]
    if not remaining:
        return None, None
    idx = random.choice(remaining)
    return idx, qa_list[idx]

def _normalize_score(val):
    """Accept 95, 95.0, '95', '95/100' -> 95, clamp to [0,100]."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        out = int(round(val))
    else:
        m = re.search(r"(100|[1-9]?\d)", str(val))
        out = int(m.group(1)) if m else None
    if out is None:
        return None
    return max(0, min(100, out))

def _clean_block(blk: str) -> str:
    """Fix common model glitches inside a JSON-looking block."""
    s = blk.strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r'(:)\s*,\s*', r'\1 ', s)  # "score":, 95 -> "score": 95
    return s

def _try_parse_obj(text: str):
    """Attempt JSON first, then literal_eval for single-quoted dicts."""
    try:
        return json.loads(text)
    except Exception:
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None

def extract_score_and_feedback(raw_text: str):
    """
    Robust extractor for LLM output:
    - Remove code fences
    - Try each non-greedy {...} block (auto-fix glitches)
    - Try parsing whole text
    - Regex fallbacks
    Returns: (score:int|None, feedback:str)
    """
    txt = raw_text.strip()
    txt = re.sub(r"^```[a-zA-Z]*\s*", "", txt)
    txt = re.sub(r"```$", "", txt)
    blocks = re.findall(r"\{.*?\}", txt, flags=re.S)
    for blk in blocks:
        cleaned = _clean_block(blk)
        obj = _try_parse_obj(cleaned)
        if isinstance(obj, dict):
            score = _normalize_score(obj.get("score"))
            feedback = obj.get("feedback")
            return score, ("" if feedback is None else str(feedback))
    obj = _try_parse_obj(_clean_block(txt))
    if isinstance(obj, dict):
        score = _normalize_score(obj.get("score"))
        feedback = obj.get("feedback")
        return score, ("" if feedback is None else str(feedback))
    m = re.search(r'"score"\s*[:=]\s*[, ]*\s*(100|[1-9]?\d)\b', txt, flags=re.I)
    if m:
        sc = int(m.group(1))
        mfb = re.search(r'"feedback"\s*[:=]\s*("?)(.+?)\1(?=[}\n\r]|$)', txt, flags=re.S)
        fb = mfb.group(2).strip() if mfb else txt
        return sc, fb
    for pat in [r"Overall\s*score\s*[:=]\s*(100|[1-9]?\d)\s*/\s*100",
                r"\b(100|[1-9]?\d)\s*/\s*100\b"]:
        m2 = re.search(pat, txt, flags=re.I)
        if m2:
            return int(m2.group(1)), txt
    subs = [int(x) for x in re.findall(r"(\d{1,3})\s*/\s*100", txt)]
    if subs:
        from statistics import mean
        return int(round(mean(subs))), txt
    return None, txt

def default_feedback_for_score(score: int | None) -> str:
    """Provide a friendly default when feedback is empty."""
    if score is None:
        return "Thanks — evaluation completed."
    if score >= 95:
        return "Excellent! Your answer matches the target almost perfectly."
    if score >= 90:
        return "Great! Your answer aligns very closely with the target."
    if score >= 85:
        return "Very good — only minor differences from the target."
    if score >= 75:
        return "Good answer with a few gaps compared to the target."
    return "Consider covering more of the key points from the target answer."

EVALUATOR_SYSTEM = """
You are a strict but fair grader for short ML exam answers.

PRIMARY GOAL: Evaluate the student's answer for SEMANTIC QUALITY — i.e., conceptual accuracy and whether the meaning matches the target answer — NOT surface wording.
Assume a separate metric (ROUGE-L) will measure lexical/structural overlap. Therefore:
- Do NOT penalize paraphrasing, reordering, or different phrasing if the meaning is correct.
- Focus on: (1) correctness of concepts/definitions, (2) completeness of key ideas, (3) precision (avoid irrelevant or incorrect claims), (4) correct use of ML terminology.
- Ignore style/grammar unless it changes meaning.

SCORING RUBRIC (0–100):
- 70%: Semantic correctness (facts & relationships are right)
- 30%: Completeness & precision (covers the essential ideas without irrelevant content)

OUTPUT: Return ONLY a single-line MINIFIED JSON object:
{"score": <integer 0-100>, "feedback": "<concise explanation>"}

Rules:
- "score" must be an integer (no '70/100', no floats).
- "feedback" should be 1–3 sentences, focusing on what is correct/missing/wrong (semantic, not wording).
- If the answer deserves 95–100, include a short positive note.
- No markdown, no bullets, no extra text before/after the JSON. Do not place a comma immediately after a colon.
""".strip()

# ---------------- SIDEBAR / DATA ----------------
with st.sidebar:
    st.markdown('<div class="sidebar-card"><h4>Settings</h4>', unsafe_allow_html=True)
    qa_path = st.text_input("Q&A JSON path", value="Q&A_db_practice.json")
    seed = st.number_input("Random seed", value=42, step=1)
    random.seed(int(seed))
    st.markdown("### Blended score weights")
    w_rouge = st.slider("ROUGE-L weight", 0.0, 1.0, 0.60, 0.05)
    w_llm   = 1.0 - w_rouge
    st.caption(f"LLM (semantic) weight: {w_llm:.2f}")
    st.caption(f"ROUGE backend: {_rouge['backend']}")
    st.markdown("</div>", unsafe_allow_html=True)

# Load data
try:
    qa_data = load_questions(qa_path)
except Exception as e:
    st.error(f"Failed to load questions: {e}")
    st.stop()

# ---------------- STATE ----------------
st.session_state.setdefault("asked", [])
st.session_state.setdefault("current_idx", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("mode", "answering")   # 'answering' | 'reviewing'
st.session_state.setdefault("last_eval", None)     # dict with last result

def ensure_current_question():
    if st.session_state.current_idx is None:
        idx, q = pick_question(qa_data, st.session_state.asked)
        if idx is None:
            return None
        st.session_state.current_idx = idx
        st.session_state.asked.append(idx)
    return qa_data[st.session_state.current_idx]

q = ensure_current_question()
if q is None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.success("All questions answered! 🎉 Review your results below.")
    if st.session_state.history:
        import pandas as pd
        cols = ["question", "score_llm", "rougeL", "score_blended", "feedback", "opinion"]
        df = pd.DataFrame(st.session_state.history)[cols]
        st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ---------------- QUESTION CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🧠 Question")
st.markdown(f"**{q['question']}**")

# Per-question widget keys => new question => fresh widgets
answer_key = f"answer_{st.session_state.current_idx}"
opinion_key = f"opinion_{st.session_state.current_idx}"

with st.form("answer_form", clear_on_submit=False):
    answer = st.text_area(
        "Your answer",
        height=200,
        key=answer_key,
        placeholder="Explain the concept clearly and concisely…"
    )
    # Stretch the submit button full width with columns
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        submitted = st.form_submit_button("Submit answer", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- EVALUATION ----------------
if submitted and st.session_state.mode == "answering":
    if not (answer or "").strip():
        st.warning("Please write an answer first.")
    else:
        with st.spinner("Evaluating your answer…"):
            llm = ChatOllama(model="llama3.2:3b", temperature=0)
            msgs = [
                SystemMessage(EVALUATOR_SYSTEM),
                HumanMessage(
                    f"Question: {q['question']}\n\n"
                    f"Target answer:\n{q['answer']}\n\n"
                    f"Student answer:\n{answer}\n\n"
                    f"Evaluate now."
                ),
            ]
            raw = llm.invoke(msgs).content
            score_llm, feedback = extract_score_and_feedback(raw)
            if not str(feedback).strip():
                feedback = default_feedback_for_score(score_llm)

            rougeL = compute_rougeL(answer, q["answer"])  # 0..1 or None
            score_blended = blend_scores(rougeL, score_llm, w_rouge, w_llm)

        st.session_state.last_eval = {
            "question": q["question"],
            "student_answer": answer,
            "target_answer": q["answer"],
            "score_llm": score_llm,
            "rougeL": rougeL,
            "score_blended": score_blended,
            "feedback": feedback,
            "opinion": None,
        }
        st.session_state.mode = "reviewing"

# ---------------- REVIEW CARD (persists across reruns) ----------------
if st.session_state.mode == "reviewing" and st.session_state.last_eval:
    ev = st.session_state.last_eval

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧮 Evaluation")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("LLM (semantic) score", f"{'N/A' if ev['score_llm'] is None else ev['score_llm']}/100")
    with m2:
        if ev["rougeL"] is None:
            st.metric("ROUGE-L", "N/A")
        else:
            st.metric("ROUGE-L", f"{ev['rougeL']:.3f} ({ev['rougeL']*100:.0f}%)")
    with m3:
        st.metric("Blended", f"{'N/A' if ev['score_blended'] is None else ev['score_blended']}/100")

    st.markdown(f"**Feedback:** {ev['feedback']}")

    ev["opinion"] = st.radio(
        "How do you rate this evaluation?",
        ["Useful", "Fair", "Too strict", "Unclear"],
        index=None,
        horizontal=True,
        key=opinion_key
    )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Save & Next question ➡️", key="next_btn", use_container_width=True):
            st.session_state.history.append(ev.copy())
            st.session_state.current_idx = None
            st.session_state.last_eval = None
            st.session_state.mode = "answering"
            st.rerun()
    with b2:
        if st.button("Edit your answer ⤺", key="edit_btn", use_container_width=True):
            st.session_state.mode = "answering"
            st.stop()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RESULTS TABLE CARD ----------------
if st.session_state.history:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Results so far")
    import pandas as pd
    cols = ["question", "score_llm", "rougeL", "score_blended", "feedback", "opinion"]
    df = pd.DataFrame(st.session_state.history)[cols]
    st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR UTIL ----------------
st.sidebar.divider()
if st.sidebar.button("Restart session"):
    for k in ["asked", "current_idx", "history", "mode", "last_eval"]:
        st.session_state.pop(k, None)
    st.rerun()
