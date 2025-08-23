FROM python:3.11-slim

# update from installing xgboost
FROM mia-infer:1.0.0

# System deps (fast, small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# update from installing xgboost
RUN pip install --no-cache-dir xgboost==2.0.3 scikit-optimize==0.9.0 shap==0.45.1 mlflow==2.14.1 pydantic==2.8.2 ruamel.yaml==0.18.6 scipy==1.11.4 


COPY main.py .

# envs set at deploy time
# ENV MODEL_BLOB_URL=...
# ENV TIMEFRAME=1d
ENV PYTHONPATH="/app"

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
