FROM python:3.12

WORKDIR /code
ENV PYTHONPATH=/code
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app ./app
COPY ./models ./models
COPY inference.py ./inference.py
COPY network.py ./network.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
