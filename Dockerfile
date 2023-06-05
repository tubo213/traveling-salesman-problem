FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN apt-get update && apt-get install -y \
    lsof

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
