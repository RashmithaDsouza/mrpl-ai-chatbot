FROM python:3.11

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y curl zstd

RUN pip install --no-cache-dir -r requirements.txt

RUN curl -fsSL https://ollama.com/install.sh | sh

EXPOSE 8501

CMD bash -c "ollama serve & sleep 5 && ollama pull phi3 && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"