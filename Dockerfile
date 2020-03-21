FROM UBUNTU:18.04

MAINTAINER Aaron W. Chen "awc33@cornell.edu"

RUN apt-get update -y && \ 
    apt-get install -y python-pip python-dev

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT ["python"]

CMD ["app.py"]