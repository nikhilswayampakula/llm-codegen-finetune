install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
ingest:
	python src/ingest_github.py --urls data/raw/sample_repos.txt --out data/processed
dataset:
	python src/build_dataset.py --src data/processed --out data/datasets/code_pairs.jsonl
train:
	python src/train_lora.py --config configs/train.yaml
api:
	uvicorn app.api:app --reload --port 8000
ui:
	streamlit run app/streamlit_app.py
test:
	pytest -q
