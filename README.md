# 🧠 Fine-Tuning LLMs Using Real-Time Code Files from GitHub to Build a Code Generation Model

A production-grade pipeline to **ingest real-time GitHub code**, curate a clean dataset, and **fine‑tune a code LLM** with **LoRA/PEFT** for **completion & generation**.  
Includes **FastAPI** inference server, **Streamlit** playground, tests, CI, Docker, and MIT license.

## 🔧 Features
- 🔄 **Ingest** public repos (URLs list) → filter languages, de-duplicate, strip boilerplate
- 🧹 **Curate** to JSONL pairs *(prompt → completion)* suitable for causal LM SFT
- 🏋️ **Train** LoRA on HuggingFace causal models (`gpt2`, `TinyLlama`, etc.) via **Transformers + PEFT + Accelerate**
- 🚀 **Serve** completions via **FastAPI** + interactive **Streamlit** UI
- 🧰 **DevOps**: Dockerfile, Makefile, GitHub Actions CI, `.gitkeep`

## ⚡ Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Ingest + build dataset
python src/ingest_github.py --urls data/raw/sample_repos.txt --out data/processed
python src/build_dataset.py --src data/processed --out data/datasets/code_pairs.jsonl

# 2) Train LoRA (small model for demo)
python src/train_lora.py --config configs/train.yaml

# 3) Serve API and UI
uvicorn app.api:app --reload --port 8000
streamlit run app/streamlit_app.py
```

## 📡 API
- `GET /health` → status + model loaded flag  
- `POST /generate` → JSON `{"prompt": "def add(a, b):"}` → returns completion

## 📂 Structure
```
llm-codegen-finetune/
├── app/ (FastAPI + Streamlit)
├── configs/
├── data/ (raw/ processed/ datasets/)
├── models/ (saved LoRA)
├── src/ (pipeline + training + infer)
├── tests/
└── DevOps: Dockerfile, Makefile, CI
```

**Author:** Nikhil Swayampakula · LinkedIn: https://www.linkedin.com/in/nikhil-swa-47b479366/ · **License:** MIT
