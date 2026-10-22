FROM python:3.12-slim
# LightGBM needs OpenMP at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
COPY configs ./configs
COPY dashboard ./dashboard
COPY tests ./tests
# Credentials are passed at run time: --env-file .env (LLM, KAGGLE_API_TOKEN) and -v ~/.kaggle:/root/.kaggle:ro
ENTRYPOINT ["python", "-m", "kagent"]
CMD ["demo"]
