FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

RUN apt update && apt install -y python3 python3-pip wget vim
RUN pip install torch tqdm numpy

# python 3.10


WORKDIR /app
COPY simsiam/ /app/simsiam/

ENTRYPOINT ["python3", "tmp.py"]