FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir \
    pydantic \
    fastapi \
    uvicorn \
    numpy \
    requests \
    openai

# Copy all files
COPY . .

# Environment variable for the OpenEnv port
ENV PORT=7860
EXPOSE 7860

# MANDATORY: The port here MUST be 7860 for Hugging Face to see the app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]