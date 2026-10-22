PY ?= python
COMP ?= titanic
LLM ?= openai

.PHONY: setup test e2e demo run dashboard docker-build docker-demo docker-run

setup:            ## install into the current virtualenv
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

test:             ## offline test suite (no keys needed)
	$(PY) -m pytest -q

e2e:              ## dashboard browser tests, video under runs/evidence/videos
	$(PY) -m playwright install chromium
	$(PY) -m pytest -m e2e --video on --output runs/evidence/videos

demo:             ## offline end-to-end run on a generated competition
	$(PY) -m kagent demo --task binary
	$(PY) -m kagent demo --task regression --target-percentile 0.05 --max-iterations 4

run:              ## real Kaggle run: make run COMP=titanic
	$(PY) -m kagent run --config configs/competitions/$(COMP).yaml --llm $(LLM)

dashboard:
	$(PY) -m streamlit run dashboard/app.py

docker-build:
	docker build -t kagent .

docker-demo:
	docker run --rm -v $(PWD)/runs:/app/runs kagent demo

docker-run:
	docker run --rm --env-file .env -v $(HOME)/.kaggle:/root/.kaggle:ro -v $(PWD)/runs:/app/runs \
		kagent run --config configs/competitions/$(COMP).yaml --llm $(LLM)
