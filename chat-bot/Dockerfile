FROM python:3.12

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

RUN pip install --upgrade pip

COPY ./requirements.txt ./

RUN pip install -r ./requirements.txt

COPY . .

CMD ["python", "main.py"]