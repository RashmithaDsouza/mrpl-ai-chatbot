FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN ollama pull phi3

EXPOSE 8501

CMD ollama serve & streamlit run App.py --server.port=8501 --server.address=0.0.0.0