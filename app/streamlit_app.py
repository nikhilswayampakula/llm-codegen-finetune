import streamlit as st
from src.infer import CodeGenService

st.set_page_config(page_title="LLM Code Generation Studio", layout="wide")
st.title("🧠 LLM Code Generation Studio (LoRA fine-tuned)")

try:
    svc = CodeGenService()
    st.success("Model loaded.")
except Exception:
    svc = None
    st.warning("No fine-tuned model found. Train first with `make train`.")

prompt = st.text_area("Prompt (e.g., function signature or docstring)", "def fibonacci(n):\n    """Return n-th Fibonacci number."""\n")
t = st.slider("Temperature", 0.0, 1.5, 0.2, 0.05)
p = st.slider("Top-p", 0.5, 1.0, 0.95, 0.01)

if st.button("Generate"):
    if svc is None:
        st.error("Model not loaded.")
    else:
        out = svc.generate(prompt, temperature=t, top_p=p)
        st.code(prompt + out, language="python")
