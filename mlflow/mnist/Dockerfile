FROM --platform=linux/amd64 ghcr.io/mlflow/mlflow:v2.2.1
COPY . /var/app
WORKDIR /var/app
RUN pip3 install -r requirements.txt
CMD ["python", "trainer.py"]

