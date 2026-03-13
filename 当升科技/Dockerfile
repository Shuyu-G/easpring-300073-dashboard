FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt dashboard_app.py README.md ./
COPY .streamlit ./.streamlit
COPY src ./src
COPY data ./data
COPY models ./models
COPY reports ./reports

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["sh", "-c", "streamlit run dashboard_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]

