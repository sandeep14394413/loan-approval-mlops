FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent .pyc files and enable stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and saved model artifacts
COPY src ./src
COPY models ./models

# Expose port
EXPOSE 5000

# Start with Gunicorn (2 workers, production-ready)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "src.app:app"]
