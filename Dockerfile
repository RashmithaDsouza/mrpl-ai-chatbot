FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y curl zstd

RUN curl -fsSL https://ollama.com/install.sh | sh

EXPOSE 10000

CMD ["bash", "start.sh"]
