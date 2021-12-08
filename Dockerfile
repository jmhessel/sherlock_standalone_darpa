# docker run -it --entrypoint bash --gpus '"device=0"' XXXX 
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

WORKDIR /sherlock

RUN apt update && apt install -y --no-install-recommends \
    unzip \
    git \
    wget \
    default-jre \
    build-essential

COPY . /sherlock

RUN pip install -r requirements.txt

CMD ["./run_all.sh"]
