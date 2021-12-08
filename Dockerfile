FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

WORKDIR /sherlock

COPY . /sherlock

RUN pip install -r requirements.txt

CMD ["./run_all.sh"]
