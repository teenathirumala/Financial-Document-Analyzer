FROM python:3.10

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y tesseract-ocr poppler-utils
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.enableCORS=false"]

