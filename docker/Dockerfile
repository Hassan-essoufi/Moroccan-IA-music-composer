
FROM python:3.10-slim

# Variables
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY . .

# Dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Ports
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
