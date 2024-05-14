FROM --platform=linux/amd64 pytorch/pytorch:latest
COPY . /var/app
WORKDIR /var/app
RUN pip3 install -r requirements.txt
CMD ["python", "trainer.py"]

