FROM python:3.11-slim

LABEL maintainer="jaisogani-ai"
LABEL project="nyaya-env"

# Hugging Face Spaces require running as a non-root user (user ID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Switch to root just to install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*
# Switch back to the non-root user
USER user

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user . $HOME/app

# Hugging Face Spaces use port 7860
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]