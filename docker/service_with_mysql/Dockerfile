FROM --platform=linux/amd64 python:3.11
WORKDIR /code

ENV CONF_HOST 0.0.0.0
ENV CONF_PORT 8000

COPY ./requirements.txt /code/requirements.txt
COPY ./app /code/app

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["python", "app/main.py"]
