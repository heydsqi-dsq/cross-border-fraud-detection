
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ /app/
COPY xgboost_fraud_model.pkl /app/
COPY feature_list.json /app/
COPY model_metadata.json /app/

# Update paths in main.py to use /app/
RUN sed -i 's|/content/fraud-api/|/app/|g' /app/main.py

# Expose port
EXPOSE 8080

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
