# TACIT: Beyond the Manual. Runs on Hugging Face Spaces (Docker), Render, Railway or any Docker host.
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
USER user
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
