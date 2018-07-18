FROM ubuntu:16.04

# Install Python.
RUN \
  apt-get update && \
  apt-get install -y python3 python3-pip



COPY prepfiles.zip /app/prepfiles.zip
COPY entity_rec/check_prep_files.py app/entity_rec/check_prep_files.py
RUN pip3 install requests

WORKDIR /app
RUN python3 entity_rec/check_prep_files.py


COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /app

CMD ["python3", "train.py"]